/**
 * Three.js rendering layer (PRD 6.2).
 *
 * Two retargeting paths behind the same `update()` interface, per the
 * PRD's own "ship the baseline, upgrade behind a stable interface"
 * philosophy (section 6.2/7, docs/avatar-upgrade.md):
 *
 *  1. PRIMITIVE SKELETON (always available, zero downloads): places a
 *     sphere at each tracked landmark and a cylinder along each bone,
 *     positioned straight from the MediaPipe world landmarks every frame.
 *     Supports TRUE per-joint visibility hiding -- if the camera can only
 *     see your upper body, the lower-body spheres/bones actually disappear
 *     rather than freezing or fading, because each is its own mesh.
 *
 *  2. RIGGED VRM + KALIDOKIT: a downloaded rigged humanoid driven by
 *     Kalidokit-solved bone rotations. A single skinned mesh can't be
 *     partially hidden the same precise way, so untracked limbs freeze at
 *     rest pose and the whole figure fades with overall visibility instead
 *     -- see riggedAvatar.js's fadeVRMByVisibility doc comment.
 *
 * Three explicit modes, set via setMode():
 *  - "mirror": live camera-driven mirroring, no rep tracking/deviation coloring
 *  - "demo": ignores the camera, plays a procedural reference rep on a loop
 *    (see demoMotion.js) so the user can watch correct form before trying it
 *  - "session": live mirroring WITH rep tracking + deviation coloring active
 *
 * Covers PRD 6.2 requirements: dual avatar (primary mirror + translucent
 * "ideal form" ghost), HSL green->red deviation coloring, key+ambient
 * lighting (Phong/Standard materials), perspective + orthographic cameras.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { loadVRMAvatar, applyPoseToVRM, tintVRM, fadeVRMByVisibility, buildDemoPose } from "./riggedAvatar.js";
import { getDemoAngles } from "./demoMotion.js";

// Bone connections using MediaPipe BlazePose indices (subset: torso + limbs).
const BONES = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso sides
  [23, 24], // hips
  [23, 25], [25, 27], [27, 31], // left leg + foot
  [24, 26], [26, 28], [28, 32], // right leg + foot
];
const JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32];

const GREEN = new THREE.Color(0x22c55e);
const RED = new THREE.Color(0xef4444);

// Maps a subset of joint landmark indices to the semantic joints used by
// deviationRules.js scores ({knee, hip, spine, ankle}), for per-sphere coloring.
const JOINT_SEMANTIC = {
  25: "knee", 26: "knee",
  23: "hip", 24: "hip",
  27: "ankle", 28: "ankle",
  11: "spine", 12: "spine",
};

function makeSkeleton(opacity, tintColor) {
  const group = new THREE.Group();
  const jointMeshes = {};
  const boneMeshes = [];

  const jointGeo = new THREE.SphereGeometry(0.035, 12, 12);
  for (const idx of JOINTS) {
    const mat = new THREE.MeshStandardMaterial({
      color: tintColor.clone(),
      transparent: true,
      opacity,
      roughness: 0.5,
      metalness: 0.05,
    });
    const mesh = new THREE.Mesh(jointGeo, mat);
    mesh.userData.baseOpacity = opacity;
    group.add(mesh);
    jointMeshes[idx] = mesh;
  }

  const boneGeo = new THREE.CylinderGeometry(0.018, 0.018, 1, 8);
  for (const [a, b] of BONES) {
    const mat = new THREE.MeshStandardMaterial({
      color: tintColor.clone(),
      transparent: true,
      opacity,
      roughness: 0.5,
      metalness: 0.05,
    });
    const mesh = new THREE.Mesh(boneGeo, mat);
    mesh.userData.baseOpacity = opacity;
    group.add(mesh);
    boneMeshes.push({ a, b, mesh });
  }

  return { group, jointMeshes, boneMeshes };
}

/**
 * World landmarks are in meters, roughly hip-centered. `mirrorX` controls
 * whether the avatar mirrors you (raise your right hand, the avatar's
 * hand on YOUR right side of the screen goes up -- like a real mirror,
 * the default) or matches your anatomical side directly (raise your right
 * hand, the avatar's own right hand goes up, which reads as "facing away
 * from you" / non-mirrored). If movement looks left/right-swapped, this
 * is the one flag to flip -- see main.js's settings panel.
 */
function landmarkToVec3(lm, mirrorX) {
  return new THREE.Vector3(mirrorX ? -lm.x : lm.x, -lm.y, -lm.z);
}

/**
 * @param {object} skeleton - {jointMeshes, boneMeshes}
 * @param {Array} worldLandmarks
 * @param {object|null} deviationScores
 * @param {object} opts
 * @param {boolean} opts.mirrorX
 * @param {Array<number>|null} opts.visibility - per-landmark 0..1 scores;
 *   joints/bones below visibilityThreshold are actually hidden (mesh
 *   .visible = false), giving true partial-body rendering -- show only
 *   your upper body to the camera, only the upper-body spheres/bones stay
 *   visible on the avatar.
 * @param {number} opts.visibilityThreshold
 */
function updateSkeleton(skeleton, worldLandmarks, deviationScores = null, opts = {}) {
  const { jointMeshes, boneMeshes } = skeleton;
  if (!worldLandmarks) return;
  const { mirrorX = true, visibility = null, visibilityThreshold = 0.5 } = opts;

  const isVisible = (idx) => !visibility || (visibility[idx] ?? 1) >= visibilityThreshold;

  for (const idx of JOINTS) {
    const lm = worldLandmarks[idx];
    const mesh = jointMeshes[idx];
    if (!lm || !isVisible(idx)) {
      mesh.visible = false;
      continue;
    }
    mesh.visible = true;
    mesh.position.copy(landmarkToVec3(lm, mirrorX));
    if (deviationScores) {
      const semantic = JOINT_SEMANTIC[idx];
      const t = semantic ? deviationScores[semantic] ?? 0 : 0;
      mesh.material.color.copy(GREEN).lerp(RED, t);
    }
  }

  for (const { a, b, mesh } of boneMeshes) {
    const pa = worldLandmarks[a];
    const pb = worldLandmarks[b];
    if (!pa || !pb || !isVisible(a) || !isVisible(b)) {
      mesh.visible = false;
      continue;
    }
    mesh.visible = true;
    const va = landmarkToVec3(pa, mirrorX);
    const vb = landmarkToVec3(pb, mirrorX);
    const mid = va.clone().add(vb).multiplyScalar(0.5);
    const dir = vb.clone().sub(va);
    const len = dir.length();

    mesh.position.copy(mid);
    mesh.scale.set(1, Math.max(len, 0.001), 1);

    const up = new THREE.Vector3(0, 1, 0);
    const quat = new THREE.Quaternion().setFromUnitVectors(up, dir.clone().normalize() || up);
    mesh.quaternion.copy(quat);

    if (deviationScores) {
      const semantic = JOINT_SEMANTIC[a] || JOINT_SEMANTIC[b];
      const t = semantic ? deviationScores[semantic] ?? 0 : 0;
      mesh.material.color.copy(GREEN).lerp(RED, t);
    }
  }
}

// ---------------------------------------------------------------------
// Demo-mode forward kinematics for the primitive skeleton: converts the
// same shared {hip, knee, spine} angle targets (demoMotion.js) that drive
// the VRM demo into raw landmark positions, using simple 2-link sagittal-
// plane kinematics (hip pivot -> knee -> ankle, hip pivot -> shoulder).
// This is a simplified illustrative stick-figure motion, not mocap --
// see demoMotion.js's doc comment.
// ---------------------------------------------------------------------
const LIMB = { thigh: 0.45, shin: 0.45, torso: 0.5, upperArm: 0.28, forearm: 0.25, stance: 0.16 };

function buildDemoLandmarks(exercise, t) {
  const { hip: hipAngle, knee: kneeAngle, spine: spineAngle, eased } = getDemoAngles(exercise, t);
  const hipY = 0.95 - eased * 0.02;
  const armSwing = 0.15 + eased * 0.1;

  const lm = new Array(33).fill(null).map(() => ({ x: 0, y: 0, z: 0, visibility: 1 }));

  const hipCenter = { x: 0, y: hipY, z: 0 };
  const shoulderCenter = {
    x: 0,
    y: hipY + LIMB.torso * Math.cos(spineAngle),
    z: LIMB.torso * Math.sin(spineAngle),
  };

  const shinAngle = Math.max(0, hipAngle - kneeAngle * 0.55);

  for (const side of [-1, 1]) {
    const hip = { x: side * LIMB.stance, y: hipCenter.y, z: hipCenter.z };
    const knee = {
      x: hip.x,
      y: hip.y - LIMB.thigh * Math.cos(hipAngle),
      z: hip.z + LIMB.thigh * Math.sin(hipAngle),
    };
    const ankle = {
      x: knee.x,
      y: knee.y - LIMB.shin * Math.cos(shinAngle),
      z: knee.z + LIMB.shin * Math.sin(shinAngle),
    };
    const foot = { x: ankle.x, y: ankle.y - 0.02, z: ankle.z + 0.16 };

    const shoulder = { x: side * LIMB.stance, y: shoulderCenter.y, z: shoulderCenter.z };
    const elbow = {
      x: shoulder.x + side * LIMB.upperArm * Math.sin(armSwing),
      y: shoulder.y - LIMB.upperArm * Math.cos(armSwing),
      z: shoulder.z,
    };
    const wrist = {
      x: elbow.x + side * LIMB.forearm * Math.sin(armSwing * 1.4),
      y: elbow.y - LIMB.forearm * Math.cos(armSwing * 1.4),
      z: elbow.z,
    };

    if (side === -1) {
      lm[23] = { ...hip, visibility: 1 };
      lm[25] = { ...knee, visibility: 1 };
      lm[27] = { ...ankle, visibility: 1 };
      lm[31] = { ...foot, visibility: 1 };
      lm[11] = { ...shoulder, visibility: 1 };
      lm[13] = { ...elbow, visibility: 1 };
      lm[15] = { ...wrist, visibility: 1 };
    } else {
      lm[24] = { ...hip, visibility: 1 };
      lm[26] = { ...knee, visibility: 1 };
      lm[28] = { ...ankle, visibility: 1 };
      lm[32] = { ...foot, visibility: 1 };
      lm[12] = { ...shoulder, visibility: 1 };
      lm[14] = { ...elbow, visibility: 1 };
      lm[16] = { ...wrist, visibility: 1 };
    }
  }

  lm[0] = { x: 0, y: shoulderCenter.y + 0.28, z: shoulderCenter.z - 0.04, visibility: 1 };

  return lm;
}

export class AvatarScene {
  constructor(containerEl, { avatarStyle = "auto", mirrorX = true, responsiveness = 0.3 } = {}) {
    this.container = containerEl;
    this.avatarStyle = avatarStyle; // "auto" | "rigged" | "skeleton"
    this.mirrorX = mirrorX;
    this.responsiveness = responsiveness;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0d10);

    const { clientWidth: w, clientHeight: h } = containerEl;

    this.perspCamera = new THREE.PerspectiveCamera(45, w / h, 0.05, 50);
    this.perspCamera.position.set(0, 1.2, 3.2);

    const orthoSize = 1.6;
    this.orthoCamera = new THREE.OrthographicCamera(
      -orthoSize * (w / h), orthoSize * (w / h), orthoSize, -orthoSize, 0.05, 50
    );
    this.orthoCamera.position.set(3.2, 1.0, 0);
    this.orthoCamera.lookAt(0, 1.0, 0);

    this.activeCamera = this.perspCamera;

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h);
    containerEl.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.perspCamera, this.renderer.domElement);
    this.controls.target.set(0, 1.0, 0);
    this.controls.enableDamping = true;

    this._setupLighting();

    this.primary = makeSkeleton(1.0, new THREE.Color(0x4f8dfd));
    this.scene.add(this.primary.group);

    this.secondary = makeSkeleton(0.4, new THREE.Color(0x22c55e));
    this.secondary.group.position.x = 1.4;
    this.scene.add(this.secondary.group);
    this.showSecondary = true;

    const grid = new THREE.GridHelper(6, 24, 0x2a2f38, 0x1a1d22);
    this.scene.add(grid);

    this.vrmPrimary = null;
    this.vrmLoadStatus = "loading"; // "loading" | "loaded" | "unavailable"
    this.clock = new THREE.Clock();
    if (avatarStyle !== "skeleton") this._loadRiggedAvatar();

    this.mode = "mirror"; // "mirror" | "demo" | "session"
    this.demoExercise = "squat";
    this.demoSpeed = 1.0;
    this._demoT = 0;

    window.addEventListener("resize", () => this.onResize());
    this._animate();
  }

  async _loadRiggedAvatar() {
    try {
      const vrm = await loadVRMAvatar();
      vrm.scene.position.set(0, 0, 0);
      this.scene.add(vrm.scene);
      this.vrmPrimary = vrm;
      this.vrmLoadStatus = "loaded";
      if (this.avatarStyle !== "skeleton") this.primary.group.visible = false;
      console.log("[FormCoach] Rigged VRM avatar loaded -- driving it via Kalidokit.");
    } catch (err) {
      this.vrmLoadStatus = "unavailable";
      console.warn("[FormCoach] Rigged VRM avatar unavailable, using primitive skeleton primary avatar instead:", err);
    }
  }

  _setupLighting() {
    const ambient = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambient);
    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(2, 4, 3);
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x88aaff, 0.3);
    fill.position.set(-3, 2, -2);
    this.scene.add(fill);
  }

  /** "rigged" | "skeleton" | "auto" (use VRM if loaded, else skeleton) */
  setAvatarStyle(style) {
    this.avatarStyle = style;
    const useVRM = this.isUsingVRM();
    if (this.vrmPrimary) this.vrmPrimary.scene.visible = useVRM;
    this.primary.group.visible = !useVRM;
    if (style === "rigged" && !this.vrmPrimary && this.vrmLoadStatus !== "loading") {
      this._loadRiggedAvatar();
    }
  }

  isUsingVRM() {
    if (this.avatarStyle === "skeleton") return false;
    if (this.avatarStyle === "rigged") return !!this.vrmPrimary;
    return !!this.vrmPrimary; // "auto"
  }

  setMirror(mirrorX) {
    this.mirrorX = mirrorX;
  }

  setResponsiveness(responsiveness) {
    this.responsiveness = responsiveness;
  }

  setCameraMode(mode) {
    this.activeCamera = mode === "ortho" ? this.orthoCamera : this.perspCamera;
    this.controls.enabled = mode !== "ortho";
  }

  /** "mirror" (default, just follow) | "demo" (ignore camera, play reference rep) | "session" (follow + track/correct) */
  setMode(mode, { exercise = "squat" } = {}) {
    this.mode = mode;
    this.demoExercise = exercise;
    this._demoT = 0;
    this.secondary.group.visible = mode !== "demo"; // no need for a ghost while the primary IS the demo
  }

  setDemoSpeed(speed) {
    this.demoSpeed = speed;
  }

  /**
   * Call once per frame from main.js's render loop.
   * @param {Array} worldLandmarks - live landmarks (ignored while mode === "demo")
   * @param {object} deviationScores - {knee,hip,spine,ankle} 0..1, only meaningful in "session" mode
   * @param {Array|null} referenceLandmarks - optional explicit ideal-form ghost pose
   * @param {object|null} riggedPose - Kalidokit.Pose.solve() output for this frame
   * @param {Array<number>|null} visibility - per-landmark 0..1 visibility scores this frame
   * @param {number} deltaSeconds - time since last frame, for demo playback speed
   */
  update(worldLandmarks, deviationScores, referenceLandmarks, riggedPose, visibility, deltaSeconds = 1 / 60) {
    const useVRM = this.isUsingVRM();

    if (this.mode === "demo") {
      this._demoT = (this._demoT + deltaSeconds * this.demoSpeed * 0.35) % 1;
      if (useVRM) {
        const demoPose = buildDemoPose(this.demoExercise, this._demoT);
        applyPoseToVRM(this.vrmPrimary, demoPose, { responsiveness: 1, visibility: null });
      } else {
        const demoLandmarks = buildDemoLandmarks(this.demoExercise, this._demoT);
        updateSkeleton(this.primary, demoLandmarks, null, { mirrorX: false });
      }
      return;
    }

    const scoresForColoring = this.mode === "session" ? deviationScores : null;

    if (this.showSecondary) {
      updateSkeleton(this.secondary, referenceLandmarks || worldLandmarks, null, { mirrorX: this.mirrorX });
    }

    if (useVRM) {
      if (riggedPose) {
        applyPoseToVRM(this.vrmPrimary, riggedPose, {
          responsiveness: this.responsiveness,
          visibility,
        });
        fadeVRMByVisibility(this.vrmPrimary, visibility);
        tintVRM(this.vrmPrimary, scoresForColoring || {});
      }
    } else {
      updateSkeleton(this.primary, worldLandmarks, scoresForColoring, {
        mirrorX: this.mirrorX,
        visibility,
      });
    }
  }

  onResize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    this.perspCamera.aspect = w / h;
    this.perspCamera.updateProjectionMatrix();

    const orthoSize = 1.6;
    this.orthoCamera.left = -orthoSize * (w / h);
    this.orthoCamera.right = orthoSize * (w / h);
    this.orthoCamera.top = orthoSize;
    this.orthoCamera.bottom = -orthoSize;
    this.orthoCamera.updateProjectionMatrix();

    this.renderer.setSize(w, h);
  }

  _animate() {
    requestAnimationFrame(() => this._animate());
    const delta = this.clock.getDelta();
    this.vrmPrimary?.update(delta);
    this.controls.update();
    this.renderer.render(this.scene, this.activeCamera);
  }
}
