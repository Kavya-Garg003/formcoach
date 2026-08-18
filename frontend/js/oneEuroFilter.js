/**
 * One Euro Filter — standard low-latency smoothing filter for noisy signals
 * (Casiez, Roussel, Vogel 2012). Cuts jitter while staying responsive to
 * fast motion, which is exactly what PRD 6.1 asks for on raw landmarks
 * before angle computation.
 *
 * One instance per scalar value. For a 3D landmark, keep one filter per
 * axis (see LandmarkSmoother below).
 */
class LowPassFilter {
  constructor() {
    this.initialized = false;
    this.value = 0;
  }
  filter(x, alpha) {
    if (!this.initialized) {
      this.value = x;
      this.initialized = true;
    } else {
      this.value = alpha * x + (1 - alpha) * this.value;
    }
    return this.value;
  }
}

export class OneEuroFilter {
  /**
   * @param {number} minCutoff - lower = more smoothing at low speed
   * @param {number} beta - higher = less lag at high speed
   * @param {number} dCutoff - cutoff for the derivative filter
   */
  constructor(minCutoff = 1.0, beta = 0.3, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xFilter = new LowPassFilter();
    this.dxFilter = new LowPassFilter();
    this.lastTime = null;
  }

  _alpha(cutoff, dt) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  filter(value, timestampMs) {
    if (this.lastTime === null) {
      this.lastTime = timestampMs;
      this.xFilter.initialized = true;
      this.xFilter.value = value;
      return value;
    }
    let dt = (timestampMs - this.lastTime) / 1000;
    if (dt <= 0) dt = 1 / 60;
    this.lastTime = timestampMs;

    const dx = (value - this.xFilter.value) / dt;
    const edx = this.dxFilter.filter(dx, this._alpha(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(edx);
    return this.xFilter.filter(value, this._alpha(cutoff, dt));
  }
}

/**
 * Smooths an array of {x,y,z} landmarks frame-to-frame, one OneEuroFilter
 * triplet per landmark index.
 */
export class LandmarkSmoother {
  constructor(numLandmarks, minCutoff = 1.0, beta = 0.4) {
    this.filters = Array.from({ length: numLandmarks }, () => ({
      x: new OneEuroFilter(minCutoff, beta),
      y: new OneEuroFilter(minCutoff, beta),
      z: new OneEuroFilter(minCutoff, beta),
    }));
  }

  smooth(landmarks, timestampMs) {
    return landmarks.map((lm, i) => {
      const f = this.filters[i];
      return {
        x: f.x.filter(lm.x, timestampMs),
        y: f.y.filter(lm.y, timestampMs),
        z: f.z.filter(lm.z, timestampMs),
        visibility: lm.visibility,
      };
    });
  }
}
