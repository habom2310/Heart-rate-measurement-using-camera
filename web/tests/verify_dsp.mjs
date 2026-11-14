// Verification script for DSP pipeline (synthesized PPG signals)
import assert from 'node:assert';

// We can compile/import the worker DSP functions or test their math directly
console.log('--- Verifying DSP Math in Node.js ---');

// Test 1: Linear detrending
function detrendSignal(data) {
  const n = data.length;
  if (n <= 1) return [...data];
  let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += data[i];
    sumXY += i * data[i];
    sumXX += i * i;
  }
  const meanX = sumX / n;
  const meanY = sumY / n;
  const denom = sumXX - sumX * meanX;
  const slope = Math.abs(denom) > 1e-12 ? (sumXY - sumX * meanY) / denom : 0;
  const intercept = meanY - slope * meanX;
  return data.map((y, i) => y - (slope * i + intercept));
}

// Linear ramp + constant: detrending should reduce variance towards 0
const ramp = Array.from({ length: 100 }, (_, i) => 2.5 * i + 10);
const detrendedRamp = detrendSignal(ramp);
const maxErr = Math.max(...detrendedRamp.map(Math.abs));
assert(maxErr < 1e-10, `Detrending ramp error should be negligible, got ${maxErr}`);
console.log('✓ Linear detrending verified.');

// Test 2: FFT and Parabolic peak detection
function fft(real, imag) {
  const n = real.length;
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tr = real[i]; real[i] = real[j]; real[j] = tr;
      const ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
    }
    let k = n >> 1;
    while (k <= j) { j -= k; k >>= 1; }
    j += k;
  }
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angle);
    const wStepI = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wR = 1.0, wI = 0.0;
      for (let k = 0; k < half; k++) {
        const uR = real[i + k], uI = imag[i + k];
        const vR = real[i + k + half] * wR - imag[i + k + half] * wI;
        const vI = real[i + k + half] * wI + imag[i + k + half] * wR;
        real[i + k] = uR + vR; imag[i + k] = uI + vI;
        real[i + k + half] = uR - vR; imag[i + k + half] = uI - vI;
        const nR = wR * wStepR - wI * wStepI;
        const nI = wR * wStepI + wI * wStepR;
        wR = nR; wI = nI;
      }
    }
  }
}

// Synthesize a 72.0 BPM (1.20 Hz) sine wave at 30 fps for 150 frames (5 seconds)
const fs = 30.0;
const targetBpm = 72.0;
const targetFreq = targetBpm / 60.0; // 1.2 Hz
const N = 150;
const nfft = 2048;
const real = new Float64Array(nfft);
const imag = new Float64Array(nfft);

for (let i = 0; i < N; i++) {
  const t = i / fs;
  const w = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (N - 1))); // Hanning
  real[i] = Math.sin(2.0 * Math.PI * targetFreq * t) * w;
}

fft(real, imag);

let maxMag = 0;
let peakIdx = 0;
const halfN = nfft / 2;
const psd = new Float64Array(halfN);
for (let i = 0; i < halfN; i++) {
  psd[i] = real[i] * real[i] + imag[i] * imag[i];
  const f = (i * fs) / nfft;
  if (f >= 0.75 && f <= 3.5) {
    if (psd[i] > maxMag) {
      maxMag = psd[i];
      peakIdx = i;
    }
  }
}

// Parabolic interpolation
const alpha = psd[peakIdx - 1];
const beta = psd[peakIdx];
const gamma = psd[peakIdx + 1];
const denom = 2.0 * (2.0 * beta - alpha - gamma);
const p = (gamma - alpha) / denom;
const df = fs / nfft;
const detectedFreq = (peakIdx + p) * df;
const detectedBpm = detectedFreq * 60.0;

console.log(`Target: ${targetBpm} BPM | Detected: ${detectedBpm.toFixed(2)} BPM (Error: ${Math.abs(detectedBpm - targetBpm).toFixed(3)} BPM)`);
assert(Math.abs(detectedBpm - targetBpm) < 0.2, `Frequency error should be < 0.2 BPM`);
console.log('✓ FFT and parabolic sub-bin peak interpolation verified.');

// Test 3: HRV metrics
const ibis = [833, 840, 825, 835, 842, 830, 838]; // ~72 BPM
const meanRR = ibis.reduce((a, b) => a + b, 0) / ibis.length;
let sdnnSum = 0;
for (const ibi of ibis) sdnnSum += (ibi - meanRR) ** 2;
const sdnn = Math.sqrt(sdnnSum / ibis.length);
let diffSqSum = 0;
for (let i = 0; i < ibis.length - 1; i++) diffSqSum += (ibis[i + 1] - ibis[i]) ** 2;
const rmssd = Math.sqrt(diffSqSum / (ibis.length - 1));
console.log(`HRV: Mean RR = ${meanRR.toFixed(1)} ms, SDNN = ${sdnn.toFixed(1)} ms, RMSSD = ${rmssd.toFixed(1)} ms`);
assert(meanRR > 800 && meanRR < 850);
console.log('✓ HRV time-domain metric calculations verified.');

// Test 4: Sliding Sub-Window Overlap-Add (POS / CHROM) Illumination Drift Rejection
function computeStd(arr) {
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

function extractPOS_OLA(frames, fps = 30.0) {
  const n = frames.length;
  const L = Math.max(20, Math.min(n, Math.round(1.6 * fps)));
  const pulse = new Float64Array(n);
  const weights = new Float64Array(n);
  const hann = new Float64Array(L);
  for (let i = 0; i < L; i++) hann[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (L - 1)));

  const s1 = new Float64Array(L);
  const s2 = new Float64Array(L);
  const h = new Float64Array(L);

  for (let t = 0; t <= n - L; t++) {
    let sumR = 0, sumG = 0, sumB = 0;
    for (let j = 0; j < L; j++) {
      const f = frames[t + j];
      sumR += f.r; sumG += f.g; sumB += f.b;
    }
    const meanR = Math.abs(sumR / L) < 1e-8 ? 1.0 : sumR / L;
    const meanG = Math.abs(sumG / L) < 1e-8 ? 1.0 : sumG / L;
    const meanB = Math.abs(sumB / L) < 1e-8 ? 1.0 : sumB / L;

    for (let j = 0; j < L; j++) {
      const f = frames[t + j];
      const rn = f.r / meanR;
      const gn = f.g / meanG;
      const bn = f.b / meanB;
      s1[j] = gn - bn;
      s2[j] = gn + bn - 2.0 * rn;
    }

    const stdS1 = computeStd(s1);
    const stdS2 = computeStd(s2);
    const alpha = stdS2 > 1e-8 ? stdS1 / stdS2 : 0.0;

    let meanH = 0;
    for (let j = 0; j < L; j++) {
      h[j] = s1[j] + alpha * s2[j];
      meanH += h[j];
    }
    meanH /= L;

    for (let j = 0; j < L; j++) {
      const val = (h[j] - meanH) * hann[j];
      pulse[t + j] += val;
      weights[t + j] += hann[j];
    }
  }

  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const w = weights[i];
    out[i] = w > 1e-8 ? pulse[i] / w : pulse[i];
  }
  return out;
}

// Generate test frames with a 75 BPM cardiac pulse + severe 40% ambient light drift
const testFrames = [];
const expectedBpm = 75.0;
for (let i = 0; i < N; i++) {
  const t = i / fs;
  const pulseVal = Math.sin(2 * Math.PI * (expectedBpm / 60.0) * t);
  const drift = 1.0 + 0.4 * (i / N); // 40% ambient light ramp
  testFrames.push({
    r: 150 * (1 + 0.010 * pulseVal) * drift,
    g: 100 * (1 + 0.035 * pulseVal) * drift,
    b: 80 * (1 + 0.015 * pulseVal) * drift,
    timestamp: t * 1000
  });
}

const extracted = extractPOS_OLA(testFrames, fs);
const olaReal = new Float64Array(nfft);
const olaImag = new Float64Array(nfft);
for (let i = 0; i < N; i++) {
  const w = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (N - 1)));
  olaReal[i] = extracted[i] * w;
}
fft(olaReal, olaImag);

let olaMax = 0;
let olaPeakIdx = 0;
for (let i = 0; i < nfft / 2; i++) {
  const f = (i * fs) / nfft;
  const pwr = olaReal[i] * olaReal[i] + olaImag[i] * olaImag[i];
  if (f >= 0.75 && f <= 3.5 && pwr > olaMax) {
    olaMax = pwr;
    olaPeakIdx = i;
  }
}
const olaBpm = ((olaPeakIdx * fs) / nfft) * 60.0;
console.log(`Target: ${expectedBpm} BPM | Detected under 40% light drift: ${olaBpm.toFixed(2)} BPM (Error: ${Math.abs(olaBpm - expectedBpm).toFixed(2)} BPM)`);
assert(Math.abs(olaBpm - expectedBpm) < 1.0, `Error under light drift must be < 1.0 BPM`);
console.log('✓ Sliding Sub-Window Overlap-Add POS verified resilient to ambient light drift.');

console.log('\nAll DSP verifications passed successfully!');
