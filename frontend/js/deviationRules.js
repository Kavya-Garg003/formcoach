/**
 * Rule-based angle-threshold deviation classifier -- PRD 6.1 & 7:
 * "your rule-based angle-threshold system is the guaranteed-working
 * baseline -- build and ship that first."
 *
 * IMPORTANT: the numeric ranges below are reasonable starting points for a
 * demo, not a validated clinical reference. Before relying on them for
 * anything beyond a class project, replace with ranges sourced directly
 * from published strength-training literature (PRD 6.1 recommends NSCA
 * guidelines / sports-science papers) and cite the source in your paper.
 */

const RANGES = {
  squat: {
    bottomKneeFlexion: { min: 60, max: 100 },      // interior knee angle at the bottom of a good rep
    spineAngleFromVertical: { min: 0, max: 45 },   // forward lean tolerance
    kneeValgusRatioMin: 0.85,                       // knee-width / ankle-width; below this = knees caving in
  },
  lunge: {
    bottomKneeFlexion: { min: 80, max: 110 },
    spineAngleFromVertical: { min: 0, max: 30 },
  },
  deadlift: {
    spineAngleFromVertical: { min: 0, max: 35 },
  },
};

function clamp01(x) {
  return Math.min(1, Math.max(0, x));
}

/**
 * @param {string} exercise - 'squat' | 'lunge' | 'deadlift'
 * @param {object} angles - output of computeJointAngles()
 * @param {number|null} kneeValgusRatio - output of estimateKneeValgusRatio()
 * @param {string} phase - current rep phase, e.g. 'bottom'
 * @returns {{deviations: string[], scores: object}}
 *   scores maps joint name -> 0..1 deviation severity, for the color lerp.
 */
export function classifyDeviations(exercise, angles, kneeValgusRatio, phase) {
  const ranges = RANGES[exercise] || RANGES.squat;
  const deviations = [];
  const scores = { knee: 0, hip: 0, spine: 0, ankle: 0 };

  if (angles.spine_angle !== null && angles.spine_angle !== undefined) {
    const max = ranges.spineAngleFromVertical?.max ?? 45;
    if (angles.spine_angle > max) {
      deviations.push("forward_lean");
      scores.spine = clamp01((angles.spine_angle - max) / 30);
    }
  }

  if (phase === "bottom" && angles.knee_flexion !== null && ranges.bottomKneeFlexion) {
    const { min, max } = ranges.bottomKneeFlexion;
    if (angles.knee_flexion > max) {
      // angle too large at the bottom == didn't bend the knee enough == shallow rep
      deviations.push("insufficient_depth");
      scores.knee = clamp01((angles.knee_flexion - max) / 30);
    } else if (angles.knee_flexion < min) {
      deviations.push("excessive_depth");
      scores.knee = clamp01((min - angles.knee_flexion) / 30);
    }
  }

  if (exercise === "squat" && kneeValgusRatio !== null && kneeValgusRatio < ranges.kneeValgusRatioMin) {
    deviations.push("knees_inward");
    scores.knee = Math.max(scores.knee, clamp01((ranges.kneeValgusRatioMin - kneeValgusRatio) / 0.3));
  }

  if (exercise === "lunge" && phase === "bottom" && kneeValgusRatio !== null && kneeValgusRatio < 0.7) {
    deviations.push("knee_passes_toe");
    scores.knee = Math.max(scores.knee, 0.6);
  }

  return { deviations, scores };
}

export { RANGES };
