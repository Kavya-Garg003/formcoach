const BASE_URL = "http://localhost:8000";

async function postJSON(path, body) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status} ${await res.text()}`);
  return res.json();
}

async function getJSON(path) {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export const api = {
  startSession: (userId, exerciseType) =>
    postJSON("/session/start", { user_id: userId, exercise_type: exerciseType }),

  logRep: (sessionId, repIndex, jointAngles, deviations, phaseDurationsMs) =>
    postJSON("/session/log_rep", {
      session_id: sessionId,
      rep_index: repIndex,
      joint_angles: jointAngles,
      deviations,
      phase_durations_ms: phaseDurationsMs,
    }),

  chat: (sessionId, userId, message) =>
    postJSON("/coach/chat", { session_id: sessionId, user_id: userId, message }),

  history: (userId) => getJSON(`/session/history/${userId}`),
};
