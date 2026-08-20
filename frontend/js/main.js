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
  // Deliberately not using browser localStorage/sessionStorage per this
  // app's constraints -- a fresh anonymous id per page load is fine for
  // a demo; swap in real auth for anything beyond that.
  return `demo-user-${Math.random().toString(36).slice(2, 8)}`;
}

const els = {
  setupOverlay: document.getElementById("setup-overlay"),
  btnEnableCam: document.getElementById("btn-enable-cam"),
  setupStatus: document.getElementById("setup-status"),
  btnStart: document.getElementById("btn-start"),
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
};

let scene, poseEstimator, smoother, repTracker;
let sessionId = null;
let sessionActive = false;
let deviationTotals = {};
let rafHandle = null;

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

async function enableCameraAndInit() {
  els.setupStatus.textContent = "Requesting camera access...";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    els.video.srcObject = stream;
    await els.video.play();
  } catch (err) {
    els.setupStatus.textContent = `Camera error: ${err.message}. Check browser permissions.`;
    return;
  }

  els.setupStatus.textContent = "Loading pose model (first load can take a few seconds)...";
  poseEstimator = new PoseEstimator();
  try {
    await poseEstimator.init({ delegate: "GPU" });
  } catch (err) {
    console.warn("GPU delegate failed, falling back to CPU:", err);
    await poseEstimator.init({ delegate: "CPU" });
  }

  smoother = new LandmarkSmoother(33, 1.0, 0.4);
  scene = new AvatarScene(els.viewport);

  els.setupOverlay.classList.add("hidden");
  startRenderLoop();
}

function startRenderLoop() {
  const loop = () => {
    rafHandle = requestAnimationFrame(loop);
    if (!poseEstimator || !els.video.videoWidth) return;

    const now = performance.now();
    const result = poseEstimator.detect(els.video, now);
    if (!result) return;

    const smoothedWorld = smoother.smooth(result.worldLandmarks, now);
    const angles = computeJointAngles(smoothedWorld);
    const valgusRatio = estimateKneeValgusRatio(smoothedWorld);

    let phase = "idle";
    let deviationResult = { deviations: [], scores: {} };

    if (sessionActive && repTracker) {
      phase = repTracker.update(angles.knee_flexion, now);
      els.repPhaseVal.textContent = phase;

      const exercise = els.exerciseSelect.value;
      deviationResult = classifyDeviations(exercise, angles, valgusRatio, phase);
      updateDeviationBanner(deviationResult.deviations);
    }

    // Only worth solving Kalidokit's rotations once the VRM avatar has
    // actually loaded -- otherwise scene.update() ignores this argument
    // and it'd be wasted work every frame.
    const riggedPose = scene.vrmPrimary ? solvePose(result.worldLandmarks, result.landmarks, els.video) : null;

    scene.update(smoothedWorld, deviationResult.scores, null, riggedPose);
  };
  loop();
}

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

async function startSession() {
  const exercise = els.exerciseSelect.value;
  try {
    const session = await api.startSession(USER_ID, exercise);
    sessionId = session.session_id;
  } catch (err) {
    alert(`Couldn't reach the backend at http://localhost:8000 -- is it running?\n\n${err.message}`);
    return;
  }

  deviationTotals = {};
  updateStatsPanel();
  els.repCountVal.textContent = "0";

  repTracker = new RepPhaseTracker({
    onRepComplete: async ({ repIndex, minKneeAngle }) => {
      els.repCountVal.textContent = String(repIndex);

      const exercise2 = els.exerciseSelect.value;
      const angles = { knee_flexion: minKneeAngle, hip_flexion: null, spine_angle: null, ankle_dorsiflexion: null };
      const { deviations } = classifyDeviations(exercise2, angles, null, "bottom");

      for (const d of deviations) {
        deviationTotals[d] = (deviationTotals[d] || 0) + 1;
      }
      updateStatsPanel();

      try {
        await api.logRep(sessionId, repIndex, angles, deviations, null);
      } catch (err) {
        console.warn("Failed to log rep to backend:", err);
      }
    },
  });
  repTracker.reset();

  sessionActive = true;
  els.btnStart.textContent = "Session Active";
  els.btnStart.disabled = true;

  chat.addMessage("assistant", `Session started for ${exercise}. I'll be watching your form -- ask me anything once you've done a few reps.`);
}

els.btnEnableCam.addEventListener("click", enableCameraAndInit);
els.btnStart.addEventListener("click", startSession);
