export type RPPGAlgorithmType = 'POS' | 'CHROM' | 'Green';
export type ROIMode = 'cheeks' | 'all' | 'forehead';

export interface RGBFrame {
  r: number;
  g: number;
  b: number;
  timestamp: number; // in milliseconds
  motionVelocity?: number; // facial landmark velocity in px/frame
}

export interface HRVMetrics {
  ibis: number[]; // ms
  meanRR: number; // ms
  sdnn: number; // ms
  rmssd: number; // ms
  pnn50: number; // %
  meanHR: number; // bpm
}

export type SignalQuality = 'Good' | 'Fair' | 'Low';

export interface HeartRateResult {
  bpm: number;
  snr: number;
  quality: SignalQuality;
  hrv: HRVMetrics;
  filteredPulse: number[];
  rawPulse: number[];
  frequencies: number[]; // Hz or BPM
  psd: number[];
  peakFreq: number; // Hz
  bufferFill: number;
  bufferCapacity: number;
  fps: number;
  motionDetected: boolean;
  motionCompensated?: boolean;
  lightCompensated?: boolean;
}

export type WorkerMessageToWorker =
  | { type: 'FRAME'; data: RGBFrame }
  | { type: 'PROCESS_BATCH'; frames: RGBFrame[] }
  | { type: 'SET_ALGORITHM'; algorithm: RPPGAlgorithmType }
  | { type: 'SET_BUFFER_SIZE'; size: number }
  | { type: 'RESET' };

export type WorkerMessageFromWorker =
  | { type: 'RESULTS'; data: HeartRateResult }
  | { type: 'STATUS'; message: string };

export interface SessionDataRecord {
  timestamp: number;
  timeSec: number;
  r: number;
  g: number;
  b: number;
  pulse: number;
  bpm: number;
  snr: number;
  quality: SignalQuality;
}
