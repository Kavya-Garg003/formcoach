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
 * pixiv/three-vrm's own MIT-licensed repo (the exact sample they use in
 * their official examples: packages/three-vrm/examples/models/
 * VRM1_Constraint_Twist_Sample.vrm), so this runs with zero extra
 * downloads. Swap in your own VRM (e.g. exported from VRoid Studio, or
 * downloaded from VRoid Hub under a compliant license -- check the
 * model's usage terms) by changing DEFAULT_VRM_URL below.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMHumanBoneName, VRMUtils } from "@pixiv/three-vrm";
import * as Kalidokit from "kalidokit";

export const DEFAULT_VRM_URL = "./avatars/sample.vrm";

/**
 * Loads a VRM model and returns the `vrm` instance (has `.scene`,
 * `.humanoid`, `.update(deltaSeconds)`).
 */
export async function loadVRMAvatar(url = DEFAULT_VRM_URL) {
  const loader = new GLTFLoader();
  loader.register((parser) => new VRMLoaderPlugin(parser));

  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error(`${url} loaded but contained no VRM data (is it a valid .vrm file?)`);

  // Recommended perf cleanup from three-vrm's own examples.
  VRMUtils.removeUnnecessaryVertices(gltf.scene);
  VRMUtils.combineSkeletons(gltf.scene);
  vrm.scene.traverse((obj) => {
    obj.frustumCulled = false;
  });

  return vrm;
}

/**
 * Runs Kalidokit's pose solver on one frame of MediaPipe output.
 * @param {Array} worldLandmarks - metric 3D landmarks (from PoseEstimator.detect())
 * @param {Array} landmarks - normalized image-space landmarks (from PoseEstimator.detect())
 * @param {HTMLVideoElement} videoEl - used by Kalidokit to correct for aspect ratio
 */
export function solvePose(worldLandmarks, landmarks, videoEl) {
  return Kalidokit.Pose.solve(worldLandmarks, landmarks, {
    runtime: "mediapipe",
    video: videoEl,
  });
}

function rigRotation(vrm, boneName, rotation = { x: 0, y: 0, z: 0 }, dampener = 1, lerpAmount = 0.3) {
  const bone = vrm.humanoid?.getNormalizedBoneNode(boneName);
  if (!bone) return;
  const euler = new THREE.Euler(rotation.x * dampener, rotation.y * dampener, rotation.z * dampener);
  const quaternion = new THREE.Quaternion().setFromEuler(euler);
  bone.quaternion.slerp(quaternion, lerpAmount);
}

function rigPosition(vrm, boneName, position = { x: 0, y: 0, z: 0 }, dampener = 1, lerpAmount = 0.3) {
  const bone = vrm.humanoid?.getNormalizedBoneNode(boneName);
  if (!bone) return;
  const vec = new THREE.Vector3(position.x * dampener, position.y * dampener, position.z * dampener);
  bone.position.lerp(vec, lerpAmount);
}

/**
 * Applies one Kalidokit-solved pose to a VRM's humanoid bones. This is the
 * standard rigging recipe from Kalidokit's own VRM examples, adapted here.
 * Leg rotations are Kalidokit's own documented "work in progress" --
 * expect them to be less precise than the arm/spine rotations, which is
 * fine for a visual mirror but is exactly why deviation *measurement*
 * (frontend/js/jointAngles.js) is computed straight from the raw
 * landmarks, not from these retargeted bone rotations.
 */
export function applyPoseToVRM(vrm, riggedPose) {
  if (!vrm || !riggedPose) return;

  rigRotation(vrm, VRMHumanBoneName.Hips, riggedPose.Hips.rotation, 0.7, 0.3);
  rigPosition(
    vrm,
    VRMHumanBoneName.Hips,
    {
      x: -riggedPose.Hips.position.x,
      y: riggedPose.Hips.position.y + 1,
      z: -riggedPose.Hips.position.z,
    },
    1,
    0.07
  );

  rigRotation(vrm, VRMHumanBoneName.Chest, riggedPose.Spine, 0.25, 0.3);
  rigRotation(vrm, VRMHumanBoneName.Spine, riggedPose.Spine, 0.45, 0.3);

  rigRotation(vrm, VRMHumanBoneName.RightUpperArm, riggedPose.RightUpperArm, 1, 0.3);
  rigRotation(vrm, VRMHumanBoneName.RightLowerArm, riggedPose.RightLowerArm, 1, 0.3);
  rigRotation(vrm, VRMHumanBoneName.LeftUpperArm, riggedPose.LeftUpperArm, 1, 0.3);
  rigRotation(vrm, VRMHumanBoneName.LeftLowerArm, riggedPose.LeftLowerArm, 1, 0.3);

  rigRotation(vrm, VRMHumanBoneName.LeftUpperLeg, riggedPose.LeftUpperLeg, 1, 0.3);
  rigRotation(vrm, VRMHumanBoneName.LeftLowerLeg, riggedPose.LeftLowerLeg, 1, 0.3);
  rigRotation(vrm, VRMHumanBoneName.RightUpperLeg, riggedPose.RightUpperLeg, 1, 0.3);
  rigRotation(vrm, VRMHumanBoneName.RightLowerLeg, riggedPose.RightLowerLeg, 1, 0.3);
}

/**
 * Coarse whole-body deviation tint: green -> red based on the single worst
 * active deviation score this frame. VRM meshes don't have the clean
 * one-material-per-joint split our primitive skeleton has (frontend/js/
 * avatarScene.js), so per-joint HSL coloring isn't a natural fit here --
 * tinting the whole figure is the honest equivalent, and the live
 * deviation banner + stats panel still carry the precise per-joint detail.
 */
const GREEN = new THREE.Color(0x22c55e);
const RED = new THREE.Color(0xef4444);

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
