import type {
  HeartRateResult,
  RGBFrame,
  RPPGAlgorithmType,
  SignalQuality,
  WorkerMessageFromWorker,
  WorkerMessageToWorker,
} from '../types/rppg';
import { extractPulse } from './algorithms';
import {
  ButterworthBandpass3rdOrder,
  calculateHRVMetrics,
  computePSD,
  detrendSignal,
  estimateHeartRateAndSNR,
} from './dsp';

let currentAlgorithm: RPPGAlgorithmType = 'POS';
let bufferCapacity = 150;
let ringBuffer: RGBFrame[] = [];

// Physiological tracking state to prevent wild jumps during motion or lighting artifacts
let lastStableBpm = 0;
let stableConfidenceCount = 0;

const filter = new ButterworthBandpass3rdOrder(0.75, 3.5, 30.0);

function estimateFs(frames: RGBFrame[]): number {
  if (frames.length < 5) return 30.0;
  const t0 = frames[0].timestamp;
  const tEnd = frames[frames.length - 1].timestamp;
  const durationSec = (tEnd - t0) / 1000.0;
  if (durationSec > 0.1) {
    const calculatedFps = (frames.length - 1) / durationSec;
    if (calculatedFps >= 10.0 && calculatedFps <= 120.0) {
      return calculatedFps;
    }
  }
  return 30.0;
}

function analyzeMotionAndLighting(frames: RGBFrame[]): {
  motionDetected: boolean;
  lightShiftDetected: boolean;
} {
  const n = frames.length;
  if (n < 5) return { motionDetected: false, lightShiftDetected: false };

  let diffSum = 0;
  let lumDiffSum = 0;
  let recentMotionVel = 0;
  let velCount = 0;

  for (let i = 1; i < n; i++) {
    const dr = Math.abs(frames[i].r - frames[i - 1].r);
    const dg = Math.abs(frames[i].g - frames[i - 1].g);
    const db = Math.abs(frames[i].b - frames[i - 1].b);
    diffSum += dr + dg + db;

    const lumPrev = 0.299 * frames[i - 1].r + 0.587 * frames[i - 1].g + 0.114 * frames[i - 1].b;
    const lumCurr = 0.299 * frames[i].r + 0.587 * frames[i].g + 0.114 * frames[i].b;
    if (lumPrev > 1.0) {
      lumDiffSum += Math.abs(lumCurr - lumPrev) / lumPrev;
    }

    if (frames[i].motionVelocity !== undefined) {
      const weight = i >= n - 10 ? 2.0 : 1.0;
      recentMotionVel += frames[i].motionVelocity! * weight;
      velCount += weight;
    }
  }

  const avgDiff = diffSum / (n - 1);
  const avgLumDiff = lumDiffSum / (n - 1);
  const avgVel = velCount > 0 ? recentMotionVel / velCount : 0;

  // Head motion detected if landmark velocity > 3.0 px/frame or average RGB differential > 9.0
  const motionDetected = avgVel > 3.0 || avgDiff > 9.0;

  // Lighting shift detected if average relative luminance change exceeds 3.5%
  const lightShiftDetected = avgLumDiff > 0.035;

  return { motionDetected, lightShiftDetected };
}

function processBuffer(): void {
  const count = ringBuffer.length;
  const fps = estimateFs(ringBuffer);
  filter.updateFs(fps);

  if (count < 30) {
    // Insufficient frames to yield reliable spectrum
    const partialRaw = extractPulse(currentAlgorithm, ringBuffer, fps);
    const result: HeartRateResult = {
      bpm: 0,
      snr: -30.0,
      quality: 'Low',
      hrv: { ibis: [], meanRR: 0, sdnn: 0, rmssd: 0, pnn50: 0, meanHR: 0 },
      filteredPulse: partialRaw,
      rawPulse: partialRaw,
      frequencies: [],
      psd: [],
      peakFreq: 0,
      bufferFill: count,
      bufferCapacity,
      fps: Math.round(fps * 10) / 10,
      motionDetected: false,
      motionCompensated: false,
      lightCompensated: false,
    };
    self.postMessage({ type: 'RESULTS', data: result } as WorkerMessageFromWorker);
    return;
  }

  // 1. Extract pulse using selected algorithm with sub-window overlap-add (POS/CHROM)
  const rawPulse = extractPulse(currentAlgorithm, ringBuffer, fps);

  // 2. Detrend signal to remove residual DC drift
  const detrended = detrendSignal(rawPulse);

  // 3. 3rd-order zero-phase Butterworth bandpass filter (0.75 - 3.5 Hz)
  const filteredPulse = filter.filtfilt(detrended);

  // 4. Compute Welch-style PSD with 2048-point Hanning window FFT
  const { freqs, psd } = computePSD(filteredPulse, fps, 2048);

  // 5. Estimate heart rate with 3-point parabolic peak interpolation and SNR (dB)
  const { bpm: rawBpm, snr, peakFreq } = estimateHeartRateAndSNR(freqs, psd, 45.0, 210.0);

  // 6. Motion and lighting analysis
  const { motionDetected, lightShiftDetected } = analyzeMotionAndLighting(ringBuffer);

  // 7. Signal Quality classification
  let quality: SignalQuality = 'Low';
  if (snr >= 6.0) {
    quality = 'Good';
  } else if (snr >= 0.0) {
    quality = 'Fair';
  } else {
    quality = 'Low';
  }

  if (motionDetected && quality === 'Good') {
    quality = 'Fair';
  }

  // 8. Motion & Light Compensation with Physiological Inertia
  let bpm = rawBpm;
  let motionCompensated = false;
  let lightCompensated = lightShiftDetected;

  if (rawBpm >= 45.0 && rawBpm <= 210.0) {
    if (snr >= 4.0 && !motionDetected) {
      if (lastStableBpm > 0 && Math.abs(rawBpm - lastStableBpm) > 12.0) {
        // Sudden jump despite high SNR - apply slight damping until sustained for multiple frames
        if (stableConfidenceCount < 3) {
          bpm = Math.round((0.75 * lastStableBpm + 0.25 * rawBpm) * 10) / 10;
          motionCompensated = true;
          stableConfidenceCount++;
        } else {
          lastStableBpm = rawBpm;
          stableConfidenceCount = 0;
        }
      } else {
        lastStableBpm = rawBpm;
        stableConfidenceCount = 0;
      }
    } else if (motionDetected || snr < 2.0) {
      // In the presence of subtle motion or low SNR, retain physiological continuity
      if (lastStableBpm > 0) {
        bpm = Math.round((0.85 * lastStableBpm + 0.15 * rawBpm) * 10) / 10;
        motionCompensated = true;
      }
    }
  }

  // 9. Calculate HRV metrics from pulse peaks
  const hrv = calculateHRVMetrics(filteredPulse, fps);

  // 10. Trim spectrum to cardiac band 40 - 220 BPM for display
  const displayFreqs: number[] = [];
  const displayPsd: number[] = [];
  for (let i = 0; i < freqs.length; i++) {
    const bpmVal = freqs[i] * 60.0;
    if (bpmVal >= 40.0 && bpmVal <= 220.0) {
      displayFreqs.push(Math.round(bpmVal * 10) / 10);
      displayPsd.push(psd[i]);
    }
  }

  const result: HeartRateResult = {
    bpm,
    snr,
    quality,
    hrv,
    filteredPulse,
    rawPulse,
    frequencies: displayFreqs,
    psd: displayPsd,
    peakFreq,
    bufferFill: count,
    bufferCapacity,
    fps: Math.round(fps * 10) / 10,
    motionDetected,
    motionCompensated,
    lightCompensated,
  };

  self.postMessage({ type: 'RESULTS', data: result } as WorkerMessageFromWorker);
}

self.onmessage = (event: MessageEvent<WorkerMessageToWorker>) => {
  const msg = event.data;
  switch (msg.type) {
    case 'FRAME': {
      ringBuffer.push(msg.data);
      if (ringBuffer.length > bufferCapacity) {
        ringBuffer.shift();
      }
      processBuffer();
      break;
    }
    case 'PROCESS_BATCH': {
      ringBuffer = msg.frames.slice(-bufferCapacity);
      processBuffer();
      break;
    }
    case 'SET_ALGORITHM': {
      currentAlgorithm = msg.algorithm;
      if (ringBuffer.length >= 30) {
        processBuffer();
      }
      break;
    }
    case 'SET_BUFFER_SIZE': {
      bufferCapacity = msg.size;
      if (ringBuffer.length > bufferCapacity) {
        ringBuffer = ringBuffer.slice(-bufferCapacity);
      }
      if (ringBuffer.length >= 30) {
        processBuffer();
      }
      break;
    }
    case 'RESET': {
      ringBuffer = [];
      lastStableBpm = 0;
      stableConfidenceCount = 0;
      processBuffer();
      break;
    }
  }
};
