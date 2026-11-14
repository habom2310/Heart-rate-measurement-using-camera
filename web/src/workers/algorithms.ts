import type { RGBFrame, RPPGAlgorithmType } from '../types/rppg';

function computeStd(arr: Float64Array | number[]): number {
  const n = arr.length;
  if (n <= 1) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += arr[i];
  const mean = sum / n;
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const diff = arr[i] - mean;
    varSum += diff * diff;
  }
  return Math.sqrt(varSum / n);
}

/**
 * Plane-Orthogonal-to-Skin (POS) Algorithm (Wang et al. 2017)
 * Enhanced with Sliding Sub-Window Overlap-Add (OLA) for illumination drift
 * and subtle motion resistance.
 */
export function extractPOS(frames: RGBFrame[], fps = 30.0): number[] {
  const n = frames.length;
  if (n === 0) return [];

  // Optimal sub-window length L ~= 1.6s (48 frames at 30 fps) as specified by Wang et al. 2017
  const L = Math.max(20, Math.min(n, Math.round(1.6 * fps)));

  // Fallback to single static window if buffer is smaller than sub-window
  if (n < L) {
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    for (let i = 0; i < n; i++) {
      sumR += frames[i].r;
      sumG += frames[i].g;
      sumB += frames[i].b;
    }
    const meanR = Math.abs(sumR / n) < 1e-8 ? 1.0 : sumR / n;
    const meanG = Math.abs(sumG / n) < 1e-8 ? 1.0 : sumG / n;
    const meanB = Math.abs(sumB / n) < 1e-8 ? 1.0 : sumB / n;

    const s1 = new Float64Array(n);
    const s2 = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const rn = frames[i].r / meanR;
      const gn = frames[i].g / meanG;
      const bn = frames[i].b / meanB;
      s1[i] = gn - bn;
      s2[i] = gn + bn - 2.0 * rn;
    }

    const stdS1 = computeStd(s1);
    const stdS2 = computeStd(s2);
    const alpha = stdS2 > 1e-8 ? stdS1 / stdS2 : 0.0;

    const pulse = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      pulse[i] = s1[i] + alpha * s2[i];
    }
    return pulse;
  }

  // Sliding Sub-Window Overlap-Add (Wang et al. 2017)
  const pulse = new Float64Array(n);
  const weights = new Float64Array(n);

  // Precompute Hanning taper window
  const hann = new Float64Array(L);
  for (let i = 0; i < L; i++) {
    hann[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (L - 1)));
  }

  const s1 = new Float64Array(L);
  const s2 = new Float64Array(L);
  const h = new Float64Array(L);

  for (let t = 0; t <= n - L; t++) {
    // 1. Local sub-window channel means (cancels slow illumination drift)
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    for (let j = 0; j < L; j++) {
      const f = frames[t + j];
      sumR += f.r;
      sumG += f.g;
      sumB += f.b;
    }
    const meanR = Math.abs(sumR / L) < 1e-8 ? 1.0 : sumR / L;
    const meanG = Math.abs(sumG / L) < 1e-8 ? 1.0 : sumG / L;
    const meanB = Math.abs(sumB / L) < 1e-8 ? 1.0 : sumB / L;

    // 2. Normalized chrominance projections
    for (let j = 0; j < L; j++) {
      const f = frames[t + j];
      const rn = f.r / meanR;
      const gn = f.g / meanG;
      const bn = f.b / meanB;
      s1[j] = gn - bn;
      s2[j] = gn + bn - 2.0 * rn;
    }

    // 3. Dynamic skin-plane projection vector alpha for this local window
    const stdS1 = computeStd(s1);
    const stdS2 = computeStd(s2);
    const alpha = stdS2 > 1e-8 ? stdS1 / stdS2 : 0.0;

    let meanH = 0;
    for (let j = 0; j < L; j++) {
      h[j] = s1[j] + alpha * s2[j];
      meanH += h[j];
    }
    meanH /= L;

    // 4. Overlap-add with Hanning window to eliminate boundary artifacts
    for (let j = 0; j < L; j++) {
      const val = (h[j] - meanH) * hann[j];
      pulse[t + j] += val;
      weights[t + j] += hann[j];
    }
  }

  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const w = weights[i];
    out[i] = w > 1e-8 ? pulse[i] / w : pulse[i];
  }
  return out;
}

/**
 * Chrominance-based (CHROM) Algorithm (de Haan & Jeanne 2013)
 * Enhanced with Sliding Sub-Window Overlap-Add (OLA) for illumination drift
 * and specular motion artifact suppression.
 */
export function extractCHROM(frames: RGBFrame[], fps = 30.0): number[] {
  const n = frames.length;
  if (n === 0) return [];

  const L = Math.max(20, Math.min(n, Math.round(1.6 * fps)));

  if (n < L) {
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    for (let i = 0; i < n; i++) {
      sumR += frames[i].r;
      sumG += frames[i].g;
      sumB += frames[i].b;
    }
    const meanR = Math.abs(sumR / n) < 1e-8 ? 1.0 : sumR / n;
    const meanG = Math.abs(sumG / n) < 1e-8 ? 1.0 : sumG / n;
    const meanB = Math.abs(sumB / n) < 1e-8 ? 1.0 : sumB / n;

    const x = new Float64Array(n);
    const y = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      const rn = frames[i].r / meanR;
      const gn = frames[i].g / meanG;
      const bn = frames[i].b / meanB;

      x[i] = 3.0 * rn - 2.0 * gn;
      y[i] = 1.5 * rn + gn - 1.5 * bn;
    }

    const stdX = computeStd(x);
    const stdY = computeStd(y);
    const alpha = stdY > 1e-8 ? stdX / stdY : 0.0;

    const pulse = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      pulse[i] = x[i] - alpha * y[i];
    }
    return pulse;
  }

  const pulse = new Float64Array(n);
  const weights = new Float64Array(n);

  const hann = new Float64Array(L);
  for (let i = 0; i < L; i++) {
    hann[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (L - 1)));
  }

  const x = new Float64Array(L);
  const y = new Float64Array(L);
  const h = new Float64Array(L);

  for (let t = 0; t <= n - L; t++) {
    let sumR = 0;
    let sumG = 0;
    let sumB = 0;
    for (let j = 0; j < L; j++) {
      const f = frames[t + j];
      sumR += f.r;
      sumG += f.g;
      sumB += f.b;
    }
    const meanR = Math.abs(sumR / L) < 1e-8 ? 1.0 : sumR / L;
    const meanG = Math.abs(sumG / L) < 1e-8 ? 1.0 : sumG / L;
    const meanB = Math.abs(sumB / L) < 1e-8 ? 1.0 : sumB / L;

    for (let j = 0; j < L; j++) {
      const f = frames[t + j];
      const rn = f.r / meanR;
      const gn = f.g / meanG;
      const bn = f.b / meanB;

      x[j] = 3.0 * rn - 2.0 * gn;
      y[j] = 1.5 * rn + gn - 1.5 * bn;
    }

    const stdX = computeStd(x);
    const stdY = computeStd(y);
    const alpha = stdY > 1e-8 ? stdX / stdY : 0.0;

    let meanH = 0;
    for (let j = 0; j < L; j++) {
      h[j] = x[j] - alpha * y[j];
      meanH += h[j];
    }
    meanH /= L;

    for (let j = 0; j < L; j++) {
      const val = (h[j] - meanH) * hann[j];
      pulse[t + j] += val;
      weights[t + j] += hann[j];
    }
  }

  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const w = weights[i];
    out[i] = w > 1e-8 ? pulse[i] / w : pulse[i];
  }
  return out;
}

/**
 * Green Channel Baseline Algorithm
 * Normalized with local temporal moving average to eliminate slow lighting drift.
 */
export function extractGreen(frames: RGBFrame[], fps = 30.0): number[] {
  const n = frames.length;
  if (n === 0) return [];

  const L = Math.max(15, Math.min(n, Math.round(1.5 * fps)));
  const pulse = new Array<number>(n);

  if (n < L) {
    let sumG = 0;
    for (let i = 0; i < n; i++) {
      sumG += frames[i].g;
    }
    const meanG = Math.abs(sumG / n) < 1e-8 ? 1.0 : sumG / n;
    for (let i = 0; i < n; i++) {
      pulse[i] = frames[i].g / meanG - 1.0;
    }
    return pulse;
  }

  // Local moving average AC/DC baseline normalization
  const halfL = Math.floor(L / 2);
  for (let i = 0; i < n; i++) {
    const start = Math.max(0, i - halfL);
    const end = Math.min(n, i + halfL + 1);
    let winSum = 0;
    for (let k = start; k < end; k++) {
      winSum += frames[k].g;
    }
    const localMean = winSum / (end - start);
    pulse[i] = Math.abs(localMean) > 1e-8 ? frames[i].g / localMean - 1.0 : 0.0;
  }

  return pulse;
}

export function extractPulse(
  algorithm: RPPGAlgorithmType,
  frames: RGBFrame[],
  fps = 30.0,
): number[] {
  switch (algorithm) {
    case 'POS':
      return extractPOS(frames, fps);
    case 'CHROM':
      return extractCHROM(frames, fps);
    case 'Green':
      return extractGreen(frames, fps);
    default:
      return extractPOS(frames, fps);
  }
}
