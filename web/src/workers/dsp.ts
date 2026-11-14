/**
 * Digital Signal Processing (DSP) library for rPPG pulse extraction.
 * Includes Butterworth filtering, detrending, Radix-2 FFT, Welch PSD,
 * parabolic sub-bin peak interpolation, SNR calculation, and HRV metrics.
 */

export interface BiquadCoeffs {
  b0: number;
  b1: number;
  b2: number;
  a1: number;
  a2: number;
}

export interface FirstOrderCoeffs {
  b0: number;
  b1: number;
  a1: number;
}

/**
 * 3rd-order Butterworth Bandpass Filter (0.75 Hz - 3.5 Hz)
 * Composed of:
 * 1) 3rd order Butterworth Highpass at lowcut (0.75 Hz) = 1st order HP + 2nd order HP biquad (Q=1)
 * 2) 3rd order Butterworth Lowpass at highcut (3.5 Hz) = 1st order LP + 2nd order LP biquad (Q=1)
 */
export class ButterworthBandpass3rdOrder {
  private fs: number;
  private lowcut: number;
  private highcut: number;

  private hp1!: FirstOrderCoeffs;
  private hp2!: BiquadCoeffs;
  private lp1!: FirstOrderCoeffs;
  private lp2!: BiquadCoeffs;

  constructor(lowcut = 0.75, highcut = 3.5, fs = 30.0) {
    this.lowcut = lowcut;
    this.highcut = highcut;
    this.fs = fs;
    this.recomputeCoefficients();
  }

  public updateFs(fs: number): void {
    if (Math.abs(this.fs - fs) > 0.1 && fs > 5.0) {
      this.fs = fs;
      this.recomputeCoefficients();
    }
  }

  private recomputeCoefficients(): void {
    const nyq = 0.5 * this.fs;
    const safeLow = Math.max(0.1, Math.min(this.lowcut, nyq * 0.45));
    const safeHigh = Math.max(safeLow + 0.1, Math.min(this.highcut, nyq * 0.95));

    // 1st order Highpass at safeLow
    const kHp = Math.tan((Math.PI * safeLow) / this.fs);
    this.hp1 = {
      b0: 1.0 / (1.0 + kHp),
      b1: -1.0 / (1.0 + kHp),
      a1: (kHp - 1.0) / (1.0 + kHp),
    };

    // 2nd order Highpass at safeLow (Butterworth stage 2, Q = 1.0)
    const w0Hp = (2.0 * Math.PI * safeLow) / this.fs;
    const cosHp = Math.cos(w0Hp);
    const alphaHp = Math.sin(w0Hp) / 2.0; // Q = 1.0 -> alpha = sin / 2
    const a0Hp = 1.0 + alphaHp;
    this.hp2 = {
      b0: (1.0 + cosHp) / (2.0 * a0Hp),
      b1: -(1.0 + cosHp) / a0Hp,
      b2: (1.0 + cosHp) / (2.0 * a0Hp),
      a1: (-2.0 * cosHp) / a0Hp,
      a2: (1.0 - alphaHp) / a0Hp,
    };

    // 1st order Lowpass at safeHigh
    const kLp = Math.tan((Math.PI * safeHigh) / this.fs);
    this.lp1 = {
      b0: kLp / (1.0 + kLp),
      b1: kLp / (1.0 + kLp),
      a1: (kLp - 1.0) / (1.0 + kLp),
    };

    // 2nd order Lowpass at safeHigh (Butterworth stage 2, Q = 1.0)
    const w0Lp = (2.0 * Math.PI * safeHigh) / this.fs;
    const cosLp = Math.cos(w0Lp);
    const alphaLp = Math.sin(w0Lp) / 2.0;
    const a0Lp = 1.0 + alphaLp;
    this.lp2 = {
      b0: (1.0 - cosLp) / (2.0 * a0Lp),
      b1: (1.0 - cosLp) / a0Lp,
      b2: (1.0 - cosLp) / (2.0 * a0Lp),
      a1: (-2.0 * cosLp) / a0Lp,
      a2: (1.0 - alphaLp) / a0Lp,
    };
  }

  /**
   * Direct Form II forward filter for 1st order section
   */
  private filter1stOrder(input: number[], c: FirstOrderCoeffs): number[] {
    const out = new Array<number>(input.length);
    let prevX = 0;
    let prevY = 0;
    for (let i = 0; i < input.length; i++) {
      const x = input[i];
      const y = c.b0 * x + c.b1 * prevX - c.a1 * prevY;
      out[i] = y;
      prevX = x;
      prevY = y;
    }
    return out;
  }

  /**
   * Direct Form II forward filter for 2nd order biquad section
   */
  private filterBiquad(input: number[], c: BiquadCoeffs): number[] {
    const out = new Array<number>(input.length);
    let x1 = 0;
    let x2 = 0;
    let y1 = 0;
    let y2 = 0;
    for (let i = 0; i < input.length; i++) {
      const x = input[i];
      const y = c.b0 * x + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
      out[i] = y;
      x2 = x1;
      x1 = x;
      y2 = y1;
      y1 = y;
    }
    return out;
  }

  /**
   * Single-pass forward cascaded filtering
   */
  public filterForward(input: number[]): number[] {
    if (input.length < 3) return [...input];
    let sig = this.filter1stOrder(input, this.hp1);
    sig = this.filterBiquad(sig, this.hp2);
    sig = this.filter1stOrder(sig, this.lp1);
    sig = this.filterBiquad(sig, this.lp2);
    return sig;
  }

  /**
   * Zero-phase forward-backward filtering (similar to scipy.signal.filtfilt)
   */
  public filtfilt(input: number[]): number[] {
    if (input.length < 6) return this.filterForward(input);

    // Forward pass
    const forward = this.filterForward(input);

    // Reverse array
    forward.reverse();

    // Backward pass
    const backward = this.filterForward(forward);

    // Reverse back
    backward.reverse();
    return backward;
  }
}

/**
 * Linear detrending (least squares linear regression removal)
 */
export function detrendSignal(data: number[]): number[] {
  const n = data.length;
  if (n <= 1) return [...data];

  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;

  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += data[i];
    sumXY += i * data[i];
    sumXX += i * i;
  }

  const meanX = sumX / n;
  const meanY = sumY / n;
  const denominator = sumXX - sumX * meanX;
  const slope = Math.abs(denominator) > 1e-12 ? (sumXY - sumX * meanY) / denominator : 0;
  const intercept = meanY - slope * meanX;

  const result = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    result[i] = data[i] - (slope * i + intercept);
  }
  return result;
}

/**
 * Radix-2 In-place Cooley-Tukey FFT. Length must be a power of 2.
 */
export function fft(real: Float64Array, imag: Float64Array): void {
  const n = real.length;
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tempR = real[i];
      real[i] = real[j];
      real[j] = tempR;
      const tempI = imag[i];
      imag[i] = imag[j];
      imag[j] = tempI;
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angle);
    const wStepI = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let wR = 1.0;
      let wI = 0.0;
      for (let k = 0; k < half; k++) {
        const uR = real[i + k];
        const uI = imag[i + k];
        const vR = real[i + k + half] * wR - imag[i + k + half] * wI;
        const vI = real[i + k + half] * wI + imag[i + k + half] * wR;

        real[i + k] = uR + vR;
        imag[i + k] = uI + vI;
        real[i + k + half] = uR - vR;
        imag[i + k + half] = uI - vI;

        const nextWR = wR * wStepR - wI * wStepI;
        const nextWI = wR * wStepI + wI * wStepR;
        wR = nextWR;
        wI = nextWI;
      }
    }
  }
}

/**
 * Compute Power Spectral Density (PSD) using Hanning windowed zero-padded FFT
 */
export function computePSD(
  signalData: number[],
  fs: number,
  nfft = 2048,
): { freqs: number[]; psd: number[] } {
  const n = signalData.length;
  if (n < 4) {
    return { freqs: [], psd: [] };
  }

  // Ensure nfft is power of 2 and >= n
  let actualNfft = 1;
  while (actualNfft < nfft || actualNfft < n) {
    actualNfft <<= 1;
  }

  const real = new Float64Array(actualNfft);
  const imag = new Float64Array(actualNfft);

  // Apply Hanning window
  let windowPowerSum = 0;
  for (let i = 0; i < n; i++) {
    const w = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (n - 1)));
    real[i] = signalData[i] * w;
    windowPowerSum += w * w;
  }

  fft(real, imag);

  const halfN = actualNfft / 2;
  const freqs = new Array<number>(halfN);
  const psd = new Array<number>(halfN);
  const scale = 1.0 / (fs * (windowPowerSum || 1.0));

  for (let i = 0; i < halfN; i++) {
    freqs[i] = (i * fs) / actualNfft;
    const magSq = real[i] * real[i] + imag[i] * imag[i];
    // Double power for positive frequencies (except DC and Nyquist)
    psd[i] = (i === 0 || i === halfN - 1 ? magSq : magSq * 2.0) * scale;
  }

  return { freqs, psd };
}

/**
 * Estimate Heart Rate (BPM) with 3-point parabolic peak interpolation
 * and compute Signal-to-Noise Ratio (SNR) in dB.
 */
export function estimateHeartRateAndSNR(
  freqs: number[],
  psd: number[],
  minBpm = 45.0,
  maxBpm = 210.0,
): { bpm: number; snr: number; peakFreq: number } {
  if (freqs.length === 0 || psd.length === 0) {
    return { bpm: 0, snr: -30.0, peakFreq: 0 };
  }

  const minF = minBpm / 60.0;
  const maxF = maxBpm / 60.0;

  let maxPsd = -Infinity;
  let peakIdx = -1;

  for (let i = 0; i < freqs.length; i++) {
    const f = freqs[i];
    if (f >= minF && f <= maxF) {
      if (psd[i] > maxPsd) {
        maxPsd = psd[i];
        peakIdx = i;
      }
    }
  }

  if (peakIdx === -1 || maxPsd <= 0) {
    return { bpm: 0, snr: -30.0, peakFreq: 0 };
  }

  // 3-point parabolic sub-bin interpolation
  let peakF = freqs[peakIdx];
  if (peakIdx > 0 && peakIdx < psd.length - 1) {
    const alpha = psd[peakIdx - 1];
    const beta = psd[peakIdx];
    const gamma = psd[peakIdx + 1];
    const denom = 2.0 * (2.0 * beta - alpha - gamma);
    if (Math.abs(denom) > 1e-12) {
      let p = (gamma - alpha) / denom;
      p = Math.max(-0.5, Math.min(0.5, p));
      const df = freqs[1] - freqs[0];
      peakF = freqs[peakIdx] + p * df;
    }
  }

  const bpm = Math.max(minBpm, Math.min(maxBpm, peakF * 60.0));

  // Calculate SNR (in dB) within physiological range
  const deltaF = 0.15; // +/- 0.15 Hz window around fundamental & harmonic
  const fFund = peakF;
  const fHarm = 2.0 * peakF;

  let pSig = 0.0;
  let pNoise = 0.0;

  for (let i = 0; i < freqs.length; i++) {
    const f = freqs[i];
    if (f >= minF && f <= maxF) {
      const isFund = Math.abs(f - fFund) <= deltaF;
      const isHarm = fHarm <= maxF && Math.abs(f - fHarm) <= deltaF;
      if (isFund || isHarm) {
        pSig += psd[i];
      } else {
        pNoise += psd[i];
      }
    }
  }

  let snr: number;
  if (pNoise <= 1e-12) {
    snr = 30.0;
  } else if (pSig <= 1e-12) {
    snr = -30.0;
  } else {
    snr = 10.0 * Math.log10(pSig / pNoise);
  }

  return {
    bpm: Math.round(bpm * 10) / 10,
    snr: Math.round(snr * 10) / 10,
    peakFreq: Math.round(peakF * 1000) / 1000,
  };
}

/**
 * Detect cardiac pulse peaks and compute time-domain HRV metrics:
 * RMSSD, SDNN, pNN50, meanRR, meanHR
 */
export function calculateHRVMetrics(
  pulseSignal: number[],
  fs: number,
): {
  ibis: number[];
  meanRR: number;
  sdnn: number;
  rmssd: number;
  pnn50: number;
  meanHR: number;
} {
  const n = pulseSignal.length;
  if (n < 5 || fs <= 0) {
    return { ibis: [], meanRR: 0, sdnn: 0, rmssd: 0, pnn50: 0, meanHR: 0 };
  }

  // Minimum distance corresponding to max 210 BPM
  const minDistance = Math.max(2, Math.floor((fs * 60.0) / 210.0));

  // Compute standard deviation
  let sum = 0;
  for (let i = 0; i < n; i++) sum += pulseSignal[i];
  const mean = sum / n;
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const diff = pulseSignal[i] - mean;
    varSum += diff * diff;
  }
  const std = Math.sqrt(varSum / n);
  const threshold = 0.15 * std;

  // Local maxima detection
  const peakIndices: number[] = [];
  let lastPeak = -minDistance;

  for (let i = 1; i < n - 1; i++) {
    if (
      pulseSignal[i] > pulseSignal[i - 1] &&
      pulseSignal[i] > pulseSignal[i + 1] &&
      pulseSignal[i] > threshold &&
      i - lastPeak >= minDistance
    ) {
      peakIndices.push(i);
      lastPeak = i;
    }
  }

  if (peakIndices.length < 2) {
    return { ibis: [], meanRR: 0, sdnn: 0, rmssd: 0, pnn50: 0, meanHR: 0 };
  }

  // Inter-Beat Intervals (IBIs) in ms
  const ibis: number[] = [];
  for (let i = 0; i < peakIndices.length - 1; i++) {
    const dt = ((peakIndices[i + 1] - peakIndices[i]) / fs) * 1000.0;
    // Physiologically realistic RR intervals: 285ms (210 BPM) - 1333ms (45 BPM)
    if (dt >= 285 && dt <= 1333) {
      ibis.push(Math.round(dt * 10) / 10);
    }
  }

  if (ibis.length === 0) {
    return { ibis: [], meanRR: 0, sdnn: 0, rmssd: 0, pnn50: 0, meanHR: 0 };
  }

  // Mean RR
  const ibiSum = ibis.reduce((a, b) => a + b, 0);
  const meanRR = ibiSum / ibis.length;

  // SDNN
  let sdnnSum = 0;
  for (const ibi of ibis) {
    const diff = ibi - meanRR;
    sdnnSum += diff * diff;
  }
  const sdnn = Math.sqrt(sdnnSum / ibis.length);

  // RMSSD & pNN50
  let rmssd = 0;
  let pnn50 = 0;
  if (ibis.length > 1) {
    let diffSqSum = 0;
    let countDiff50 = 0;
    for (let i = 0; i < ibis.length - 1; i++) {
      const diff = ibis[i + 1] - ibis[i];
      diffSqSum += diff * diff;
      if (Math.abs(diff) > 50) {
        countDiff50++;
      }
    }
    rmssd = Math.sqrt(diffSqSum / (ibis.length - 1));
    pnn50 = (countDiff50 / (ibis.length - 1)) * 100.0;
  }

  const meanHR = meanRR > 0 ? 60000.0 / meanRR : 0;

  return {
    ibis,
    meanRR: Math.round(meanRR * 10) / 10,
    sdnn: Math.round(sdnn * 10) / 10,
    rmssd: Math.round(rmssd * 10) / 10,
    pnn50: Math.round(pnn50 * 10) / 10,
    meanHR: Math.round(meanHR * 10) / 10,
  };
}
