/**
 * Three.js rendering layer (PRD 6.2).
 *
 * MVP retargeting choice: rather than driving a rigged Mixamo/ReadyPlayerMe
 * glTF through Kalidokit IK (PRD's Week 6-8 stretch goal), this direct-
 * mapping "primitive skeleton" avatar places a sphere at each tracked
 * landmark and a cylinder along each bone, positioned straight from the
 * MediaPipe world landmarks every frame. It has no rig to fight with, so
 * it's the guaranteed-working baseline described in PRD 6.2/7 -- get this
 * running first, then swap in Kalidokit + a real rigged model behind the
 * same `update(landmarks)` interface once that's working (see docs/avatar-upgrade.md).
 *
 * Covers PRD 6.2 requirements:
 *  - dual avatar (primary mirror + translucent "ideal form" ghost)
 *  - HSL green->red deviation coloring per joint
 *  - key light + ambient (Phong/Standard material lighting model)
 *  - perspective (orbit) + orthographic (side "form check") cameras
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

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
      transparent: opacity < 1,
      opacity,
      roughness: 0.5,
      metalness: 0.05,
    });
    const mesh = new THREE.Mesh(jointGeo, mat);
    group.add(mesh);
    jointMeshes[idx] = mesh;
  }

  const boneGeo = new THREE.CylinderGeometry(0.018, 0.018, 1, 8);
  for (const [a, b] of BONES) {
    const mat = new THREE.MeshStandardMaterial({
      color: tintColor.clone(),
      transparent: opacity < 1,
      opacity,
      roughness: 0.5,
      metalness: 0.05,
    });
    const mesh = new THREE.Mesh(boneGeo, mat);
    group.add(mesh);
    boneMeshes.push({ a, b, mesh });
  }

  return { group, jointMeshes, boneMeshes };
}

/**
 * Repositions joints/bones from MediaPipe world landmarks. World landmarks
 * are in meters, roughly hip-centered, X right / Y up / Z toward camera --
 * we flip X for a natural mirror and flip Y/Z sign conventions to match
 * Three.js's right-handed Y-up world.
 */
function landmarkToVec3(lm) {
  return new THREE.Vector3(-lm.x, -lm.y, -lm.z);
}

function updateSkeleton({ jointMeshes, boneMeshes }, worldLandmarks, deviationScores = null) {
  if (!worldLandmarks) return;

  for (const idx of JOINTS) {
    const lm = worldLandmarks[idx];
    if (!lm) continue;
    jointMeshes[idx].position.copy(landmarkToVec3(lm));
    if (deviationScores) {
      const semantic = JOINT_SEMANTIC[idx];
      const t = semantic ? deviationScores[semantic] ?? 0 : 0;
      jointMeshes[idx].material.color.copy(GREEN).lerp(RED, t);
    }
  }

  for (const { a, b, mesh } of boneMeshes) {
    const pa = worldLandmarks[a];
    const pb = worldLandmarks[b];
    if (!pa || !pb) continue;
    const va = landmarkToVec3(pa);
    const vb = landmarkToVec3(pb);
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

export class AvatarScene {
  constructor(containerEl) {
    this.container = containerEl;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0d10);

    const { clientWidth: w, clientHeight: h } = containerEl;

    // Perspective camera + orbit controls -- default live-mirroring view.
    this.perspCamera = new THREE.PerspectiveCamera(45, w / h, 0.05, 50);
    this.perspCamera.position.set(0, 1.2, 3.2);

    // Orthographic camera -- side-view "technical form-check" mode (PRD 6.2).
    const orthoSize = 1.6;
    this.orthoCamera = new THREE.OrthographicCamera(
      -orthoSize * (w / h), orthoSize * (w / h), orthoSize, -orthoSize, 0.05, 50
    );
    this.orthoCamera.position.set(3.2, 1.0, 0); // side-on
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

    // Primary avatar: mirrors the user live.
    this.primary = makeSkeleton(1.0, new THREE.Color(0x4f8dfd));
    this.scene.add(this.primary.group);

    // Secondary avatar: translucent "ideal form" reference, offset to the side.
    this.secondary = makeSkeleton(0.4, new THREE.Color(0x22c55e));
    this.secondary.group.position.x = 1.4;
    this.scene.add(this.secondary.group);

    const grid = new THREE.GridHelper(6, 24, 0x2a2f38, 0x1a1d22);
    this.scene.add(grid);

    window.addEventListener("resize", () => this.onResize());
    this._animate();
  }

  _setupLighting() {
    // Ambient + directional key light using MeshStandardMaterial, per PRD 6.2's
    // explicit call-out of the CGVR illumination-model module.
    const ambient = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambient);

    const key = new THREE.DirectionalLight(0xffffff, 1.2);
    key.position.set(2, 4, 3);
    this.scene.add(key);

    const fill = new THREE.DirectionalLight(0x88aaff, 0.3);
    fill.position.set(-3, 2, -2);
    this.scene.add(fill);
  }

  setCameraMode(mode) {
    this.activeCamera = mode === "ortho" ? this.orthoCamera : this.perspCamera;
    this.controls.enabled = mode !== "ortho";
  }

  /**
   * @param {Array} worldLandmarks - primary (live) mirrored avatar
   * @param {object} deviationScores - {knee,hip,spine,ankle} 0..1
   * @param {Array|null} referenceLandmarks - optional ideal-form ghost pose;
   *   falls back to mirroring the same live pose (untinted by deviation) if omitted.
   */
  update(worldLandmarks, deviationScores, referenceLandmarks = null) {
    updateSkeleton(this.primary, worldLandmarks, deviationScores);
    updateSkeleton(this.secondary, referenceLandmarks || worldLandmarks, null);
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
    this.controls.update();
    this.renderer.render(this.scene, this.activeCamera);
  }
}
