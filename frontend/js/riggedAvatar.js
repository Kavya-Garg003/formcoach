/**
 * Rigged VRM avatar driven by Kalidokit (PRD 6.2's Week 6-8 retargeting
 * milestone, per docs/avatar-upgrade.md). This is the "real" retargeting
 * path -- frontend/js/avatarScene.js's primitive sphere-and-cylinder
 * skeleton is the guaranteed-working fallback if this one fails to load
 * (missing file, network blocked, WebGL issue, etc.), per the PRD's own
 * "ship the baseline, upgrade behind the same interface" philosophy.
 *
 * Why VRM specifically, not a raw Mixamo glTF: Kalidokit's Pose.solve()
 * output uses VRM's standard humanoid bone names directly (Hips, Spine,
 * LeftUpperArm, RightUpperLeg, ...), and @pixiv/three-vrm resolves those
 * names against *any* VRM model's actual internal skeleton for you via
 * `vrm.humanoid.getNormalizedBoneNode(name)`. That avoids hand-writing a
 * bone-name mapping table per downloaded model, which is the annoying
 * part of Mixamo-based retargeting.
 *
 * Included sample model: frontend/avatars/sample.vrm, fetched from
 * pixiv/three-vrm's own MIT-licensed repo. Swap in your own VRM (e.g.
 * exported from VRoid Studio) by changing DEFAULT_VRM_URL below.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMHumanBoneName, VRMUtils } from "@pixiv/three-vrm";
import * as Kalidokit from "kalidokit";
import { getDemoAngles } from "./demoMotion.js";

export const DEFAULT_VRM_URL = "./avatars/sample.vrm";

// Which BlazePose landmark indices "back" each humanoid bone, used to
// decide whether a bone should freeze at rest instead of following stale/
// noisy tracking when that part of the body isn't actually visible to the
// camera. See applyPoseToVRM's `visibility` param.
const BONE_SOURCE_LANDMARKS = {
  [VRMHumanBoneName.Hips]: [23, 24],
  [VRMHumanBoneName.Chest]: [11, 12],
  [VRMHumanBoneName.Spine]: [11, 12, 23, 24],
  [VRMHumanBoneName.RightUpperArm]: [12, 14],
  [VRMHumanBoneName.RightLowerArm]: [14, 16],
  [VRMHumanBoneName.LeftUpperArm]: [11, 13],
  [VRMHumanBoneName.LeftLowerArm]: [13, 15],
  [VRMHumanBoneName.LeftUpperLeg]: [23, 25],
  [VRMHumanBoneName.LeftLowerLeg]: [25, 27],
  [VRMHumanBoneName.RightUpperLeg]: [24, 26],
  [VRMHumanBoneName.RightLowerLeg]: [26, 28],
};

/** Loads a VRM model and returns the `vrm` instance. */
export async function loadVRMAvatar(url = DEFAULT_VRM_URL) {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));

  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error(`${url} loaded but contained no VRM data (is it a valid .vrm file?)`);

  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  vrm.scene.traverse((obj) => {
    obj.frustumCulled = false;
  });

  return vrm;
}

/**
 * Runs Kalidokit's pose solver on one frame of MediaPipe output.
 * @param {Array} worldLandmarks - metric 3D landmarks
 * @param {Array} landmarks - normalized image-space landmarks
 * @param {HTMLVideoElement} videoEl - used by Kalidokit to correct for aspect ratio
 */
export function solvePose(worldLandmarks, landmarks, videoEl) {
  return Kalidokit.Pose.solve(worldLandmarks, landmarks, {
    runtime: "mediapipe",
    video: videoEl,
  });
}

function boneVisible(boneName, visibility, threshold) {
  if (!visibility) return true; // no visibility data supplied (e.g. demo mode) -- always drive it
  const sourceIdx = BONE_SOURCE_LANDMARKS[boneName];
  if (!sourceIdx) return true;
  return sourceIdx.every((i) => (visibility[i] ?? 1) >= threshold);
}

function rigRotation(vrm, boneName, rotation = { x: 0, y: 0, z: 0 }, dampener, lerpAmount, visibility, visThreshold) {
  const bone = vrm.humanoid?.getNormalizedBoneNode(boneName);
  if (!bone) return;

  if (!boneVisible(boneName, visibility, visThreshold)) {
    // Part of the body the camera can't currently see -- ease back toward
    // the model's neutral rest pose instead of holding a stale rotation or
    // following noisy low-confidence tracking. This is the practical
    // equivalent, for a single skinned VRM mesh, of "only show what's
    // actually tracked" -- true partial-mesh hiding isn't possible without
    // per-region vertex masking, but freezing untracked limbs at rest
    // gives the same "I can only see your upper body" read.
    const restQuat = bone.userData._restQuaternion;
    if (restQuat) bone.quaternion.slerp(restQuat, 0.15);
    return;
  }
  if (!bone.userData._restQuaternion) {
    bone.userData._restQuaternion = bone.quaternion.clone();
  }

  const euler = new THREE.Euler(rotation.x * dampener, rotation.y * dampener, rotation.z * dampener);
  const quaternion = new THREE.Quaternion().setFromEuler(euler);
  bone.quaternion.slerp(quaternion, lerpAmount);
}

function rigPosition(vrm, boneName, position = { x: 0, y: 0, z: 0 }, dampener, lerpAmount) {
  const bone = vrm.humanoid?.getNormalizedBoneNode(boneName);
  if (!bone) return;
  const vec = new THREE.Vector3(position.x * dampener, position.y * dampener, position.z * dampener);
  bone.position.lerp(vec, lerpAmount);
}

/**
 * Applies one Kalidokit-solved pose to a VRM's humanoid bones.
 * @param {object} vrm
 * @param {object} riggedPose - Kalidokit.Pose.solve() output, or a
 *   synthetic pose from buildDemoPose() below.
 * @param {object} opts
 * @param {number} opts.responsiveness - 0..1, how quickly bones snap to
 *   the target rotation each frame. Higher = catches subtle/fast movement
 *   better but looks jitterier; lower = smoother but laggy on small
 *   motions. Default 0.3 was tuned for "looks smooth", not "misses
 *   nothing" -- raise this (try 0.5-0.6) if you need it to pick up subtle
 *   movement, not just big joint swings.
 * @param {Array<number>|null} opts.visibility - per-landmark 0..1
 *   visibility scores for this frame (from MediaPipe), or null to always
 *   drive every bone regardless of visibility (used by demo playback).
 * @param {number} opts.visibilityThreshold - landmarks below this are
 *   treated as "not currently visible to the camera".
 */
export function applyPoseToVRM(vrm, riggedPose, opts = {}) {
  if (!vrm || !riggedPose) return;
  const { responsiveness = 0.3, visibility = null, visibilityThreshold = 0.5 } = opts;

  rigRotation(vrm, VRMHumanBoneName.Hips, riggedPose.Hips.rotation, 0.7, responsiveness, visibility, visibilityThreshold);
  rigPosition(
    vrm,
    VRMHumanBoneName.Hips,
    {
      x: -riggedPose.Hips.position.x,
      y: riggedPose.Hips.position.y + 1,
      z: -riggedPose.Hips.position.z,
    },
    1,
    Math.min(0.5, responsiveness * 0.25)
  );

  rigRotation(vrm, VRMHumanBoneName.Chest, riggedPose.Spine, 0.25, responsiveness, visibility, visibilityThreshold);
  rigRotation(vrm, VRMHumanBoneName.Spine, riggedPose.Spine, 0.45, responsiveness, visibility, visibilityThreshold);

  rigRotation(vrm, VRMHumanBoneName.RightUpperArm, riggedPose.RightUpperArm, 1, responsiveness, visibility, visibilityThreshold);
  rigRotation(vrm, VRMHumanBoneName.RightLowerArm, riggedPose.RightLowerArm, 1, responsiveness, visibility, visibilityThreshold);
  rigRotation(vrm, VRMHumanBoneName.LeftUpperArm, riggedPose.LeftUpperArm, 1, responsiveness, visibility, visibilityThreshold);
  rigRotation(vrm, VRMHumanBoneName.LeftLowerArm, riggedPose.LeftLowerArm, 1, responsiveness, visibility, visibilityThreshold);

  rigRotation(vrm, VRMHumanBoneName.LeftUpperLeg, riggedPose.LeftUpperLeg, 1, responsiveness, visibility, visibilityThreshold);
  rigRotation(vrm, VRMHumanBoneName.LeftLowerLeg, riggedPose.LeftLowerLeg, 1, responsiveness, visibility, visibilityThreshold);
  rigRotation(vrm, VRMHumanBoneName.RightUpperLeg, riggedPose.RightUpperLeg, 1, responsiveness, visibility, visibilityThreshold);
  rigRotation(vrm, VRMHumanBoneName.RightLowerLeg, riggedPose.RightLowerLeg, 1, responsiveness, visibility, visibilityThreshold);
}

/**
 * Fades the whole VRM figure's opacity based on what fraction of landmarks
 * are currently visible -- e.g. only your upper body in frame reads as
 * "camera can only see your top half" via a faded lower body, since a
 * single skinned mesh can't be split into independently-hideable regions
 * the way the primitive skeleton's separate joint/bone meshes can (see
 * avatarScene.js -- that path gives TRUE partial-body hiding; this is the
 * closest honest equivalent for a rigged mesh).
 */
export function fadeVRMByVisibility(vrm, visibility) {
  if (!vrm || !visibility) return;
  const visibleFraction = visibility.filter((v) => (v ?? 0) >= 0.5).length / visibility.length;
  const opacity = 0.35 + 0.65 * visibleFraction; // never fully invisible, floor at 35%

  vrm.scene.traverse((obj) => {
    if (obj.isMesh && obj.material) {
      const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const mat of materials) {
        mat.transparent = true;
        mat.opacity = opacity;
      }
    }
  });
}

const GREEN = new THREE.Color(0x22c55e);
const RED = new THREE.Color(0xef4444);

/**
 * Coarse whole-body deviation tint: green -> red based on the single worst
 * active deviation score this frame. See module comment on why this is
 * whole-figure rather than per-joint for a rigged mesh.
 */
export function tintVRM(vrm, deviationScores = {}) {
  if (!vrm) return;
  const maxScore = Math.max(0, ...Object.values(deviationScores));
  const tint = GREEN.clone().lerp(RED, maxScore);

  vrm.scene.traverse((obj) => {
    if (obj.isMesh && obj.material) {
      const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const mat of materials) {
        if (mat.color) {
          if (!mat.userData._baseColor) mat.userData._baseColor = mat.color.clone();
          mat.color.copy(mat.userData._baseColor).lerp(tint, maxScore * 0.6);
        }
      }
    }
  });
}

// ---------------------------------------------------------------------
// Demo mode: converts the shared {hip, knee, spine} angle targets from
// demoMotion.js into a synthetic pose shaped like Kalidokit's Pose.solve()
// output, so demo playback can go through the exact same applyPoseToVRM()
// as live tracking does -- one rigging code path, not two.
// ---------------------------------------------------------------------

/**
 * @param {string} exercise - 'squat' | 'lunge' | 'deadlift'
 * @param {number} t - 0..1, position within one rep cycle.
 */
export function buildDemoPose(exercise, t) {
  const { hip, knee, spine, eased } = getDemoAngles(exercise, t);

  return {
    Hips: {
      position: { x: 0, y: -eased * 0.15, z: 0 },
      rotation: { x: spine * 0.3, y: 0, z: 0 },
    },
    Spine: { x: spine, y: 0, z: 0 },
    LeftUpperLeg: { x: hip, y: 0, z: 0.05 },
    RightUpperLeg: { x: hip, y: 0, z: -0.05 },
    LeftLowerLeg: { x: -knee, y: 0, z: 0 },
    RightLowerLeg: { x: -knee, y: 0, z: 0 },
    LeftUpperArm: { x: 0.15, y: 0, z: 1.3 },
    RightUpperArm: { x: 0.15, y: 0, z: -1.3 },
    LeftLowerArm: { x: 0, y: 0, z: 0.1 },
    RightLowerArm: { x: 0, y: 0, z: -0.1 },
  };
}
