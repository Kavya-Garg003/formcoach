import { PoseEstimator } from "./poseEstimation.js";
import { LandmarkSmoother } from "./oneEuroFilter.js";
import { computeJointAngles, estimateKneeValgusRatio } from "./jointAngles.js";
import { RepPhaseTracker } from "./repPhase.js";
import { classifyDeviations } from "./deviationRules.js";
import { AvatarScene } from "./avatarScene.js";
import { solvePose } from "./riggedAvatar.js";
import { ChatPanel } from "./chat.js";
import { api } from "./api.js";

const USER_ID = localStorage_getUserId();

function localStorage_getUserId() {
  // Deliberately not using browser localStorage/sessionStorage -- a fresh
  // anonymous id per page load is fine for a demo; swap in real auth for
  // anything beyond that.
  return `demo-user-${Math.random().toString(36).slice(2, 8)}`;
}

const els = {
  setupOverlay: document.getElementById("setup-overlay"),
  btnEnableCam: document.getElementById("btn-enable-cam"),
  setupStatus: document.getElementById("setup-status"),
  btnStart: document.getElementById("btn-start"),
  btnStopSession: document.getElementById("btn-stop-session"),
  btnBackToMirror: document.getElementById("btn-back-to-mirror"),
  modeLabel: document.getElementById("mode-label"),
  exerciseSelect: document.getElementById("exercise"),
  video: document.getElementById("webcam"),
  viewport: document.getElementById("three-viewport"),
  repCountVal: document.getElementById("rep-count-val"),
  repPhaseVal: document.getElementById("rep-phase-val"),
  statsBody: document.getElementById("stats-body"),
  deviationBanner: document.getElementById("live-deviation-banner"),
  camPerspBtn: document.getElementById("btn-cam-perspective"),
  camOrthoBtn: document.getElementById("btn-cam-ortho"),
  chatLog: document.getElementById("chat-log"),
  chatForm: document.getElementById("chat-form"),
  chatInput: document.getElementById("chat-input"),
  btnSettingsToggle: document.getElementById("btn-settings-toggle"),
  settingsPanel: document.getElementById("settings-panel"),
  settingAvatarStyle: document.getElementById("setting-avatar-style"),
  settingMirror: document.getElementById("setting-mirror"),
  settingResponsiveness: document.getElementById("setting-responsiveness"),
  settingDemoSpeed: document.getElementById("setting-demo-speed"),
  settingCameraDevice: document.getElementById("setting-camera-device"),
  debugDelegate: document.getElementById("debug-delegate"),
  debugFps: document.getElementById("debug-fps"),
  debugVrm: document.getElementById("debug-vrm"),
  debugVisibility: document.getElementById("debug-visibility"),
};

let scene, poseEstimator, smoother, repTracker;
let sessionId = null;
let appMode = "mirror"; // "mirror" | "demo" | "session"
let deviationTotals = {};
let currentStream = null;

// Adjustable via the settings panel -- see wireSettingsPanel() below.
const settings = {
  mirrorX: true,
  responsiveness: 0.35,
  demoSpeed: 1.0,
  avatarStyle: "auto",
  visibilityThreshold: 0.5,
};

// Calibration: rather than assuming a fixed 170deg "standing" knee angle
// for everyone, sample the user's own actual standing angle for the first
// ~25 frames of a session and use that instead -- adapts to body
// proportions and camera angle instead of one hardcoded number for everyone.
let calibration = { active: false, samples: [], onDone: null };

// Buffers this rep's raw landmarks so they can be sent to the trained
// classifier (if one is loaded -- see backend/app/classifier_infer.py)
// once the rep completes.
let currentRepFrames = [];

// FPS tracking for the debug status bar.
let fpsFrameCount = 0;
let fpsLastSampleTime = performance.now();

const chat = new ChatPanel({
  logEl: els.chatLog,
  formEl: els.chatForm,
  inputEl: els.chatInput,
  getSessionId: () => sessionId,
  getUserId: () => USER_ID,
});

els.camPerspBtn.addEventListener("click", () => setCameraMode("persp"));
els.camOrthoBtn.addEventListener("click", () => setCameraMode("ortho"));

function setCameraMode(mode) {
  scene?.setCameraMode(mode);
  els.camPerspBtn.classList.toggle("active", mode === "persp");
  els.camOrthoBtn.classList.toggle("active", mode === "ortho");
}

// ---------------------------------------------------------------------
// Camera device enumeration (for phone-as-webcam apps like Iriun/DroidCam)
// ---------------------------------------------------------------------
async function refreshCameraDeviceList() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === "videoinput");
    const previousValue = els.settingCameraDevice.value;
    els.settingCameraDevice.innerHTML = '<option value="">Default camera</option>';
    cams.forEach((cam, i) => {
      const opt = document.createElement("option");
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `Camera ${i + 1}`;
      els.settingCameraDevice.appendChild(opt);
    });
    if (cams.some((c) => c.deviceId === previousValue)) {
      els.settingCameraDevice.value = previousValue;
    }
  } catch (err) {
    console.warn("[FormCoach] Couldn't list camera devices:", err);
  }
}

async function openCameraStream(deviceId) {
  const videoConstraints = deviceId
    ? { deviceId: { exact: deviceId }, width: 640, height: 480 }
    : { width: 640, height: 480, facingMode: "user" };

  currentStream?.getTracks().forEach((t) => t.stop());
  const stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
  currentStream = stream;
  els.video.srcObject = stream;
  await els.video.play();
}

// ---------------------------------------------------------------------
// Setup / camera enable
// ---------------------------------------------------------------------
async function enableCameraAndInit() {
  els.setupStatus.textContent = "Requesting camera access...";
  try {
    await openCameraStream(els.settingCameraDevice.value || null);
  } catch (err) {
    els.setupStatus.textContent = `Camera error: ${err.message}. Check browser permissions.`;
    return;
  }

  await refreshCameraDeviceList(); // labels are only populated after permission is granted

  els.setupStatus.textContent = "Loading pose model (first load can take a few seconds)...";
  poseEstimator = new PoseEstimator();
  try {
    await poseEstimator.init({ delegate: "GPU" });
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU:", err);
    await poseEstimator.init({ delegate: "CPU" });
  }
  els.debugDelegate.textContent = `delegate: ${poseEstimator.activeDelegate}`;
  if (poseEstimator.activeDelegate === "CPU") {
    console.warn(
      "[FormCoach] Running on the CPU delegate -- this is commonly why tracking " +
      "looks slow/laggy. GPU delegate failed to initialize (see the warning above " +
      "this one for why); a low-power GPU, an outdated browser, or a blocked WebGL " +
      "context are the usual causes."
    );
  }

  smoother = new LandmarkSmoother(33, 1.0, 0.4);
  scene = new AvatarScene(els.viewport, {
    avatarStyle: settings.avatarStyle,
    mirrorX: settings.mirrorX,
    responsiveness: settings.responsiveness,
  });

  // Create the backend session NOW, not gated behind "Start Session" --
  // this is what makes chat usable immediately instead of silently doing
  // nothing if you try to chat before starting a rep-tracking session.
  try {
    const exercise = els.exerciseSelect.value;
    const session = await api.startSession(USER_ID, exercise);
    sessionId = session.session_id;
    chat.setEnabled(true);
    chat.addMessage(
      "assistant",
      `Hi! I'm watching your form. Pick an exercise above to see a demo of correct form, or hit "Start Session" whenever you're ready and I'll track and correct you live.`
    );
  } catch (err) {
    chat.addMessage("assistant", `(Couldn't reach the backend at http://localhost:8000 to start a session -- is uvicorn running? Chat and rep logging won't work until it is. Error: ${err.message})`);
  }

  els.btnStart.disabled = false;
  els.btnStart.title = "";
  els.setupOverlay.classList.add("hidden");
  startRenderLoop();
}

// ---------------------------------------------------------------------
// Mode switching: mirror (default) <-> demo (exercise selected) <-> session
// ---------------------------------------------------------------------
function setAppMode(mode, { exercise } = {}) {
  appMode = mode;
  scene?.setMode(mode, { exercise: exercise || els.exerciseSelect.value });

  els.modeLabel.textContent = mode === "mirror" ? "Mirror" : mode === "demo" ? "Demo" : "Session";
  els.modeLabel.className = mode;

  els.btnBackToMirror.classList.toggle("hidden", mode !== "demo");
  els.btnStart.classList.toggle("hidden", mode === "session");
  els.btnStopSession.classList.toggle("hidden", mode !== "session");
  els.exerciseSelect.disabled = mode === "session";

  if (mode !== "session") {
    els.deviationBanner.classList.add("hidden");
    els.repPhaseVal.textContent = "idle";
  }
}

els.exerciseSelect.addEventListener("change", () => {
  if (appMode === "session") return; // don't yank the demo out from under an active session
  if (!scene) return; // camera not enabled yet -- nothing to demo
  setAppMode("demo", { exercise: els.exerciseSelect.value });
});

els.btnBackToMirror.addEventListener("click", () => setAppMode("mirror"));

els.btnStopSession.addEventListener("click", () => {
  repTracker = null;
  setAppMode("mirror");
});

// ---------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------
function startRenderLoop() {
  let lastTime = performance.now();

  const loop = () => {
    requestAnimationFrame(loop);
    if (!poseEstimator || !els.video.videoWidth) return;

    const now = performance.now();
    const deltaSeconds = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;
    updateFpsCounter(now);

    if (appMode === "demo") {
      // AvatarScene drives its own procedural playback in demo mode and
      // ignores landmarks entirely -- see avatarScene.js's update().
      scene.update(null, null, null, null, null, deltaSeconds);
      return;
    }

    const result = poseEstimator.detect(els.video, now);
    if (!result) return;

    const smoothedWorld = smoother.smooth(result.worldLandmarks, now);
    const visibility = smoothedWorld.map((lm) => lm.visibility ?? 1);
    updateVisibilityDebug(visibility);

    const angles = computeJointAngles(smoothedWorld);
    const valgusRatio = estimateKneeValgusRatio(smoothedWorld);

    let phase = "idle";
    let deviationResult = { deviations: [], scores: {} };

    if (appMode === "session") {
      if (calibration.active) {
        runCalibrationStep(angles.knee_flexion);
      } else if (repTracker) {
        phase = repTracker.update(angles.knee_flexion, now);
        els.repPhaseVal.textContent = phase;
        currentRepFrames.push(result.worldLandmarks);
        if (currentRepFrames.length > 300) currentRepFrames.shift(); // safety cap (~10s at 30fps)

        const exercise = els.exerciseSelect.value;
        deviationResult = classifyDeviations(exercise, angles, valgusRatio, phase);
        updateDeviationBanner(deviationResult.deviations);
      }
    }

    const riggedPose = scene.vrmPrimary ? solvePose(result.worldLandmarks, result.landmarks, els.video) : null;
    scene.update(smoothedWorld, deviationResult.scores, null, riggedPose, visibility, deltaSeconds);

    els.debugVrm.textContent = `avatar: ${scene.isUsingVRM() ? "rigged VRM" : "skeleton"} (${scene.vrmLoadStatus})`;
  };
  loop();
}

function updateFpsCounter(now) {
  fpsFrameCount++;
  if (now - fpsLastSampleTime >= 500) {
    const fps = Math.round((fpsFrameCount * 1000) / (now - fpsLastSampleTime));
    els.debugFps.textContent = `fps: ${fps}`;
    fpsFrameCount = 0;
    fpsLastSampleTime = now;
  }
}

function updateVisibilityDebug(visibility) {
  const visibleCount = visibility.filter((v) => v >= settings.visibilityThreshold).length;
  els.debugVisibility.textContent = `tracked: ${visibleCount}/${visibility.length} landmarks`;
}

// ---------------------------------------------------------------------
// Calibration: measure the user's actual standing knee angle for ~25
// frames before starting real rep tracking, instead of assuming 170deg
// fits everyone.
// ---------------------------------------------------------------------
function runCalibrationStep(kneeAngle) {
  if (kneeAngle === null || kneeAngle === undefined) return;
  calibration.samples.push(kneeAngle);
  els.repPhaseVal.textContent = `calibrating (${calibration.samples.length}/25)... stand still`;
  if (calibration.samples.length >= 25) {
    const avg = calibration.samples.reduce((s, v) => s + v, 0) / calibration.samples.length;
    calibration.active = false;
    calibration.onDone(avg);
  }
}

// ---------------------------------------------------------------------
// Deviation banner / stats panel
// ---------------------------------------------------------------------
function updateDeviationBanner(deviations) {
  if (deviations.length === 0) {
    els.deviationBanner.classList.add("hidden");
    return;
  }
  els.deviationBanner.classList.remove("hidden");
  els.deviationBanner.textContent = deviations.map(prettifyLabel).join(" · ");
}

function prettifyLabel(label) {
  return label.replace(/_/g, " ");
}

function updateStatsPanel() {
  const entries = Object.entries(deviationTotals);
  if (entries.length === 0) {
    els.statsBody.textContent = "No deviations logged yet -- looking good.";
    return;
  }
  els.statsBody.innerHTML = entries
    .sort((a, b) => b[1] - a[1])
    .map(([label, count]) => `<div class="dev-row"><span>${prettifyLabel(label)}</span><span>${count}</span></div>`)
    .join("");
}

// ---------------------------------------------------------------------
// Start Session
// ---------------------------------------------------------------------
async function startSession() {
  const exercise = els.exerciseSelect.value;

  // Always create a fresh backend session on each "Start Session" click so
  // rep counts are not shared across separate workout attempts.
  try {
    const session = await api.startSession(USER_ID, exercise);
    sessionId = session.session_id;
    chat.setEnabled(true);
  } catch (err) {
    alert(`Couldn't reach the backend at http://localhost:8000 -- is it running?\n\n${err.message}`);
    return;
  }

  deviationTotals = {};
  updateStatsPanel();
  els.repCountVal.textContent = "0";
  currentRepFrames = [];

  setAppMode("session", { exercise });

  repTracker = null;
  calibration = {
    active: true,
    samples: [],
    onDone: (calibratedStandingAngle) => {
      repTracker = new RepPhaseTracker({
        standingAngle: calibratedStandingAngle,
        onRepComplete: handleRepComplete,
      });
      repTracker.reset();
      chat.addMessage(
        "assistant",
        `Calibrated to your standing knee angle (~${calibratedStandingAngle.toFixed(0)}°). Tracking your ${exercise} reps now -- go ahead.`
      );
    },
  };
}

async function handleRepComplete({ repIndex, minKneeAngle }) {
  els.repCountVal.textContent = String(repIndex);

  const exercise = els.exerciseSelect.value;
  // Include only non-null angles so Pydantic (dict[str, float]) accepts the payload.
  // Null values cause a silent 422 that prevents reps from being saved to the backend.
  const rawAngles = { knee_flexion: minKneeAngle, hip_flexion: null, spine_angle: null, ankle_dorsiflexion: null };
  const angles = Object.fromEntries(Object.entries(rawAngles).filter(([, v]) => v != null));
  const { deviations } = classifyDeviations(exercise, rawAngles, null, "bottom");

  for (const d of deviations) {
    deviationTotals[d] = (deviationTotals[d] || 0) + 1;
  }
  updateStatsPanel();

  const repFrames = currentRepFrames;
  currentRepFrames = [];

  try {
    await api.logRep(sessionId, repIndex, angles, deviations, null);
  } catch (err) {
    console.warn("Failed to log rep to backend:", err);
  }

  // Optional: ask the trained classifier for a second opinion on this rep
  // (see backend/app/classifier_infer.py). Silently skipped if no
  // classifier has been trained/loaded -- this is an enhancement on top
  // of the rule-based check, not a requirement.
  if (repFrames.length >= 5) {
    try {
      const framesPayload = repFrames.map((frame) => frame.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z })));
      const result = await api.classifyRep(framesPayload, exercise);
      if (result.available) {
        chat.addMessage(
          "assistant",
          `AI classifier's read on rep ${repIndex}: ${result.predicted_label} (${(result.confidence * 100).toFixed(0)}% confidence). ${result.caveat || ""}`
        );
      }
    } catch (err) {
      console.warn("Classifier call failed (non-fatal):", err);
    }
  }
}

// ---------------------------------------------------------------------
// Settings panel
// ---------------------------------------------------------------------
function wireSettingsPanel() {
  els.btnSettingsToggle.addEventListener("click", () => {
    els.settingsPanel.classList.toggle("hidden");
  });

  els.settingAvatarStyle.addEventListener("change", (e) => {
    settings.avatarStyle = e.target.value;
    scene?.setAvatarStyle(e.target.value);
  });

  els.settingMirror.addEventListener("change", (e) => {
    settings.mirrorX = e.target.checked;
    scene?.setMirror(e.target.checked);
  });

  els.settingResponsiveness.addEventListener("input", (e) => {
    settings.responsiveness = Number(e.target.value) / 100;
    scene?.setResponsiveness(settings.responsiveness);
  });

  els.settingDemoSpeed.addEventListener("input", (e) => {
    settings.demoSpeed = Number(e.target.value) / 100;
    scene?.setDemoSpeed(settings.demoSpeed);
  });

  els.settingCameraDevice.addEventListener("change", async (e) => {
    if (!currentStream) return; // camera not enabled yet -- picked up on enable instead
    try {
      await openCameraStream(e.target.value || null);
    } catch (err) {
      alert(`Couldn't switch camera: ${err.message}`);
    }
  });
}

els.btnEnableCam.addEventListener("click", enableCameraAndInit);
els.btnStart.addEventListener("click", startSession);
wireSettingsPanel();
refreshCameraDeviceList(); // best-effort before permission is granted -- labels fill in after
