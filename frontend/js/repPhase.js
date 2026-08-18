/**
 * Simple state machine driven by hip/knee angle + angular velocity, per
 * PRD 6.1: "setup -> descent -> bottom -> ascent". A full ascent back to
 * near the setup angle counts as one completed rep.
 *
 * Angle convention: knee_flexion here is the *interior* knee angle, so it
 * is ~170-180 deg when standing (leg straight) and gets smaller as the
 * lifter descends into a squat/lunge.
 */

const PHASES = { IDLE: "idle", SETUP: "setup", DESCENT: "descent", BOTTOM: "bottom", ASCENT: "ascent" };

export class RepPhaseTracker {
  /**
   * @param {object} opts
   * @param {number} opts.standingAngle - knee angle (deg) considered "standing"
   * @param {number} opts.descentTriggerDelta - degrees below standing angle that counts as starting a descent
   * @param {number} opts.velocityDeadband - deg/sec below which motion counts as "still" (for bottom detection)
   * @param {function} opts.onRepComplete - callback(repSummary)
   */
  constructor({ standingAngle = 170, descentTriggerDelta = 8, velocityDeadband = 15, onRepComplete = () => {} } = {}) {
    this.standingAngle = standingAngle;
    this.descentTriggerDelta = descentTriggerDelta;
    this.velocityDeadband = velocityDeadband;
    this.onRepComplete = onRepComplete;

    this.phase = PHASES.IDLE;
    this.lastAngle = null;
    this.lastTime = null;
    this.repIndex = 0;
    this.phaseStartTime = null;
    this.phaseDurations = {};
    this.minAngleThisRep = Infinity;
  }

  reset() {
    this.phase = PHASES.SETUP;
    this.lastAngle = null;
    this.lastTime = null;
    this.phaseStartTime = performance.now();
    this.phaseDurations = {};
    this.minAngleThisRep = Infinity;
  }

  /**
   * @param {number} kneeAngle - current smoothed knee flexion angle (deg)
   * @param {number} timestampMs
   * @returns {string} current phase
   */
  update(kneeAngle, timestampMs) {
    if (kneeAngle === null || kneeAngle === undefined) return this.phase;
    if (this.phase === PHASES.IDLE) this.reset();

    let velocity = 0;
    if (this.lastAngle !== null && this.lastTime !== null) {
      const dt = Math.max((timestampMs - this.lastTime) / 1000, 1e-3);
      velocity = (kneeAngle - this.lastAngle) / dt; // deg/sec, negative while descending
    }
    this.lastAngle = kneeAngle;
    this.lastTime = timestampMs;
    this.minAngleThisRep = Math.min(this.minAngleThisRep, kneeAngle);

    const nextPhase = this._transition(kneeAngle, velocity);
    if (nextPhase !== this.phase) {
      const now = timestampMs;
      const elapsed = this.phaseStartTime ? now - this.phaseStartTime : 0;
      this.phaseDurations[this.phase] = (this.phaseDurations[this.phase] || 0) + elapsed;
      this.phaseStartTime = now;
      this.phase = nextPhase;
    }

    return this.phase;
  }

  _transition(angle, velocity) {
    switch (this.phase) {
      case PHASES.SETUP:
        if (angle < this.standingAngle - this.descentTriggerDelta && velocity < 0) {
          return PHASES.DESCENT;
        }
        return PHASES.SETUP;

      case PHASES.DESCENT:
        if (Math.abs(velocity) < this.velocityDeadband) {
          return PHASES.BOTTOM;
        }
        return PHASES.DESCENT;

      case PHASES.BOTTOM:
        if (velocity > this.velocityDeadband) {
          return PHASES.ASCENT;
        }
        return PHASES.BOTTOM;

      case PHASES.ASCENT:
        if (angle >= this.standingAngle - this.descentTriggerDelta / 2) {
          this._completeRep();
          return PHASES.SETUP;
        }
        return PHASES.ASCENT;

      default:
        return PHASES.SETUP;
    }
  }

  _completeRep() {
    this.repIndex += 1;
    this.onRepComplete({
      repIndex: this.repIndex,
      minKneeAngle: this.minAngleThisRep,
      phaseDurationsMs: { ...this.phaseDurations },
    });
    this.minAngleThisRep = Infinity;
    this.phaseDurations = {};
  }
}

export { PHASES };
