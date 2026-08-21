/**
 * Shared "what does a demo rep look like" data for the Demo/Teach mode.
 * Both avatar renderers (the primitive skeleton in avatarScene.js, and
 * the rigged VRM in riggedAvatar.js) read the same {hip, knee, spine}
 * angle targets from here and convert them into their own representation
 * (forward-kinematics landmark positions vs. bone Euler rotations), so
 * there's exactly one place to tune "what correct form looks like" for
 * the demo playback instead of two copies drifting apart.
 *
 * These are simplified, hand-authored illustrative targets -- NOT motion-
 * captured or derived from the biomechanics literature. If you want a
 * more accurate reference demo, the more defensible source is the same
 * angle ranges already in frontend/js/deviationRules.js's RANGES object
 * (which the PRD asks you to source from real strength-training
 * literature) -- these keyframes are deliberately close to those ranges'
 * midpoints, but adjust freely.
 */

export const DEMO_KEYFRAMES = {
  squat: {
    standing: { hip: 0, knee: 0, spine: 0.05 },
    bottom: { hip: 0.95, knee: 1.9, spine: 0.35 },
  },
  lunge: {
    standing: { hip: 0, knee: 0, spine: 0.05 },
    bottom: { hip: 0.8, knee: 1.7, spine: 0.25 },
  },
  deadlift: {
    standing: { hip: 0, knee: 0.15, spine: 0.05 },
    bottom: { hip: 1.1, knee: 0.35, spine: 0.55 },
  },
};

function lerpNum(a, b, t) {
  return a + (b - a) * t;
}

/**
 * @param {string} exercise - 'squat' | 'lunge' | 'deadlift'
 * @param {number} t - 0..1 position within one rep cycle (0 = standing,
 *   0.5 = bottom, 1 = back to standing).
 * @returns {{hip:number, knee:number, spine:number, eased:number}} angle
 *   targets in radians, plus the raw 0..1 eased progress value in case a
 *   renderer wants it directly (e.g. for hip-height offset).
 */
export function getDemoAngles(exercise, t) {
  const kf = DEMO_KEYFRAMES[exercise] || DEMO_KEYFRAMES.squat;
  const tri = t < 0.5 ? t * 2 : (1 - t) * 2; // 0 -> 1 -> 0 triangle wave across the cycle
  const eased = Math.sin((tri * Math.PI) / 2); // ease in/out instead of a linear robotic motion

  return {
    hip: lerpNum(kf.standing.hip, kf.bottom.hip, eased),
    knee: lerpNum(kf.standing.knee, kf.bottom.knee, eased),
    spine: lerpNum(kf.standing.spine, kf.bottom.spine, eased),
    eased,
  };
}
