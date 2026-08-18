/**
 * Joint-angle engine (PRD 6.1 + Appendix).
 *
 * MediaPipe BlazePose landmark indices used here:
 *   11 L shoulder   12 R shoulder
 *   13 L elbow      14 R elbow
 *   15 L wrist      16 R wrist
 *   23 L hip        24 R hip
 *   25 L knee       26 R knee
 *   27 L ankle      28 R ankle
 *   31 L foot index 32 R foot index
 *
 * angle = arccos( (v1 . v2) / (|v1| |v2|) )   -- the exact formula from the PRD appendix.
 */

const LM = {
  L_SHOULDER: 11, R_SHOULDER: 12,
  L_ELBOW: 13, R_ELBOW: 14,
  L_WRIST: 15, R_WRIST: 16,
  L_HIP: 23, R_HIP: 24,
  L_KNEE: 25, R_KNEE: 26,
  L_ANKLE: 27, R_ANKLE: 28,
  L_FOOT_INDEX: 31, R_FOOT_INDEX: 32,
};

function sub(a, b) {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}
function dot(a, b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
function mag(a) {
  return Math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z);
}
function mid(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 };
}

/** Angle (degrees) at vertex `b`, formed by rays b->a and b->c. */
export function angleAtVertex(a, b, c) {
  const v1 = sub(a, b);
  const v2 = sub(c, b);
  const m1 = mag(v1);
  const m2 = mag(v2);
  if (m1 === 0 || m2 === 0) return null;
  const cosTheta = Math.min(1, Math.max(-1, dot(v1, v2) / (m1 * m2)));
  return (Math.acos(cosTheta) * 180) / Math.PI;
}

/** Angle (degrees) between a vector and the world "up" vertical axis. */
function angleFromVertical(vec) {
  const up = { x: 0, y: 1, z: 0 };
  const m = mag(vec);
  if (m === 0) return null;
  const cosTheta = Math.min(1, Math.max(-1, dot(vec, up) / m));
  return (Math.acos(cosTheta) * 180) / Math.PI;
}

/**
 * Computes the tracked angle set from PRD 6.1: knee flexion, hip flexion,
 * spine angle vs vertical, ankle dorsiflexion. Averages left/right when
 * both sides are visible; otherwise uses whichever side is confidently
 * tracked.
 *
 * @param {Array<{x:number,y:number,z:number,visibility?:number}>} lm - 33 world landmarks
 * @returns {{knee_flexion:number, hip_flexion:number, spine_angle:number, ankle_dorsiflexion:number, sides:object}}
 */
export function computeJointAngles(lm, visibilityThreshold = 0.5) {
  const visible = (i) => !lm[i] || lm[i].visibility === undefined || lm[i].visibility >= visibilityThreshold;

  const sideAngles = { left: {}, right: {} };

  // Knee flexion: angle at knee between hip->knee and ankle->knee
  if (visible(LM.L_HIP) && visible(LM.L_KNEE) && visible(LM.L_ANKLE)) {
    sideAngles.left.knee_flexion = angleAtVertex(lm[LM.L_HIP], lm[LM.L_KNEE], lm[LM.L_ANKLE]);
  }
  if (visible(LM.R_HIP) && visible(LM.R_KNEE) && visible(LM.R_ANKLE)) {
    sideAngles.right.knee_flexion = angleAtVertex(lm[LM.R_HIP], lm[LM.R_KNEE], lm[LM.R_ANKLE]);
  }

  // Hip flexion: angle at hip between shoulder->hip and knee->hip
  if (visible(LM.L_SHOULDER) && visible(LM.L_HIP) && visible(LM.L_KNEE)) {
    sideAngles.left.hip_flexion = angleAtVertex(lm[LM.L_SHOULDER], lm[LM.L_HIP], lm[LM.L_KNEE]);
  }
  if (visible(LM.R_SHOULDER) && visible(LM.R_HIP) && visible(LM.R_KNEE)) {
    sideAngles.right.hip_flexion = angleAtVertex(lm[LM.R_SHOULDER], lm[LM.R_HIP], lm[LM.R_KNEE]);
  }

  // Ankle dorsiflexion: angle at ankle between knee->ankle and footIndex->ankle
  if (visible(LM.L_KNEE) && visible(LM.L_ANKLE) && visible(LM.L_FOOT_INDEX)) {
    sideAngles.left.ankle_dorsiflexion = angleAtVertex(lm[LM.L_KNEE], lm[LM.L_ANKLE], lm[LM.L_FOOT_INDEX]);
  }
  if (visible(LM.R_KNEE) && visible(LM.R_ANKLE) && visible(LM.R_FOOT_INDEX)) {
    sideAngles.right.ankle_dorsiflexion = angleAtVertex(lm[LM.R_KNEE], lm[LM.R_ANKLE], lm[LM.R_FOOT_INDEX]);
  }

  // Spine angle vs vertical: vector from mid-hip to mid-shoulder
  let spineAngle = null;
  if (visible(LM.L_SHOULDER) && visible(LM.R_SHOULDER) && visible(LM.L_HIP) && visible(LM.R_HIP)) {
    const shoulderMid = mid(lm[LM.L_SHOULDER], lm[LM.R_SHOULDER]);
    const hipMid = mid(lm[LM.L_HIP], lm[LM.R_HIP]);
    spineAngle = angleFromVertical(sub(shoulderMid, hipMid));
  }

  const avg = (key) => {
    const vals = [sideAngles.left[key], sideAngles.right[key]].filter((v) => v !== undefined && v !== null);
    if (vals.length === 0) return null;
    return vals.reduce((s, v) => s + v, 0) / vals.length;
  };

  return {
    knee_flexion: avg("knee_flexion"),
    hip_flexion: avg("hip_flexion"),
    spine_angle: spineAngle,
    ankle_dorsiflexion: avg("ankle_dorsiflexion"),
    sides: sideAngles,
  };
}

/**
 * Rough frontal-plane knee-valgus proxy: compares knee-to-knee distance
 * against ankle-to-knee distance, normalized by hip width. This is a
 * placeholder heuristic, not a validated biomechanical measurement --
 * replace with the EC3D-trained classifier (PRD 6.1) for a real signal.
 */
export function estimateKneeValgusRatio(lm) {
  const hipWidth = mag(sub(lm[LM.L_HIP], lm[LM.R_HIP]));
  const kneeWidth = mag(sub(lm[LM.L_KNEE], lm[LM.R_KNEE]));
  const ankleWidth = mag(sub(lm[LM.L_ANKLE], lm[LM.R_ANKLE]));
  if (hipWidth === 0 || ankleWidth === 0) return null;
  // Ideal: knees track roughly in line with ankles. Ratio << 1 means knees
  // have drifted inward relative to the ankle stance width.
  return kneeWidth / ankleWidth;
}

export { LM };
