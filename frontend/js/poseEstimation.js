/**
 * Thin wrapper around MediaPipe Tasks Vision's PoseLandmarker (BlazePose
 * GHUM 3D, 33 landmarks), per PRD 6.1. Loaded from a CDN by default so the
 * project runs with zero local downloads; see README "What to download"
 * for how to self-host the wasm bundle + .task model file for offline use
 * or if your network blocks the CDN.
 */

const CDN_VISION_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const CDN_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// If you download these locally (see README), point these two constants at
// e.g. "./models/wasm" and "./models/pose_landmarker_lite.task" instead.
const VISION_WASM_PATH = CDN_VISION_WASM;
const MODEL_ASSET_PATH = CDN_MODEL_URL;

let PoseLandmarker, FilesetResolver;

async function loadMediapipeModule() {
  if (PoseLandmarker && FilesetResolver) return;
  const mod = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs"
  );
  PoseLandmarker = mod.PoseLandmarker;
  FilesetResolver = mod.FilesetResolver;
}

export class PoseEstimator {
  constructor() {
    this.landmarker = null;
    this.running = false;
  }

  async init({ delegate = "GPU" } = {}) {
    await loadMediapipeModule();
    const vision = await FilesetResolver.forVisionTasks(VISION_WASM_PATH);
    this.landmarker = await PoseLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: MODEL_ASSET_PATH,
        delegate,
      },
      runningMode: "VIDEO",
      numPoses: 1,
    });
  }

  /**
   * @param {HTMLVideoElement} video
   * @param {number} timestampMs
   * @returns {{landmarks: Array, worldLandmarks: Array} | null}
   *   `landmarks` are normalized [0,1] image-space coords (good for overlay).
   *   `worldLandmarks` are metric 3D coords in meters, centered at the hips
   *   (use these for joint-angle math -- they're not affected by camera distance).
   */
  detect(video, timestampMs) {
    if (!this.landmarker) return null;
    const result = this.landmarker.detectForVideo(video, timestampMs);
    if (!result.landmarks?.length) return null;
    return {
      landmarks: result.landmarks[0],
      worldLandmarks: result.worldLandmarks[0],
    };
  }

  close() {
    this.landmarker?.close();
    this.landmarker = null;
  }
}
