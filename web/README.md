# rPPG Web Application

Client-side real-time remote photoplethysmography (rPPG) web application built with **React 19**, **TypeScript**, **Tailwind CSS**, and **Vite**. Measures heart rate (BPM), signal-to-noise ratio (SNR), and heart rate variability (HRV) metrics directly in the browser using the client webcam with zero server roundtrips.

## Features

- **MediaPipe Face Mesh Tracking**: Runs `@mediapipe/tasks-vision` FaceLandmarker client-side in WebAssembly to detect 478 3D facial landmarks.
- **Biometric ROI Extraction**: Dynamically calculates precise convex hull polygon masks for the forehead and left/right malar cheek regions, excluding eyes, eyebrows, and lips to avoid motion artifacts.
- **Web Worker DSP Pipeline**: Signal processing runs on a dedicated Web Worker thread, keeping the main UI thread running smoothly at 60 FPS:
  - **Plane-Orthogonal-to-Skin (POS)**: Robust projection algorithm invariant to skin tone and head movements.
  - **Chrominance-based (CHROM)**: Standardized color difference method under varying white-balance conditions.
  - **Green Channel Mean**: Fast baseline photoplethysmography based on hemoglobin green light absorption.
- **Real-Time Telemetry & Visualizations**:
  - Live blood volume pulse (BVP) waveform rendering.
  - Welch Power Spectral Density (PSD) spectrum with cardiac band (0.7–3.5 Hz) peak detection.
  - Vital telemetry cards: Heart Rate (BPM), Confidence SNR (dB), and HRV indicators (RMSSD, SDNN, pNN50).
- **Eulerian Color Magnification (EVM)**: Real-time pulse flushing overlay to amplify subtle facial color variations.
- **Data Export**: Export complete session telemetry (time, raw RGB, pulse wave, BPM, SNR, HRV) to CSV and JSON formats.
- **Synthetic Demo Mode**: Built-in physiological signal generator for testing and demonstration when a camera is unavailable.

## Project Structure

```
web/
├── public/                 # Static assets (favicons, icons)
├── src/
│   ├── components/         # React UI components
│   │   ├── CameraFeed.tsx  # Webcam capture, MediaPipe tracking & ROI overlay
│   │   ├── SpectrumChart.tsx # Welch PSD frequency spectrum chart
│   │   ├── Toolbar.tsx     # Algorithm selection, buffer size & display toggles
│   │   ├── VitalCards.tsx  # Real-time metrics HUD (BPM, SNR, HRV)
│   │   └── WaveformChart.tsx # Real-time BVP pulse waveform visualizer
│   ├── types/
│   │   └── rppg.ts         # Shared TypeScript interfaces & message types
│   ├── utils/
│   │   └── export.ts       # CSV and JSON telemetry export utilities
│   ├── workers/
│   │   ├── algorithms.ts   # POS, CHROM, and Green-mean rPPG algorithms
│   │   ├── dsp.ts          # Bandpass filtering, detrending, Welch PSD, HRV
│   │   └── rppgWorker.ts   # Web Worker message handler & sliding buffer
│   ├── App.tsx             # Main dashboard container & Worker coordinator
│   ├── index.css           # Tailwind CSS directives & theme variables
│   └── main.tsx            # React application root
├── tests/
│   └── verify_dsp.mjs      # Node.js DSP test suite for filter, PSD & algorithms
├── package.json
├── tsconfig.json
└── vite.config.ts
```

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org) (v18 or higher recommended)
- `npm` or equivalent package manager

### Installation

```bash
cd web
npm install
```

### Development Server

Start the local Vite development server:

```bash
npm run dev
```

Open `http://localhost:5173` in your browser. Allow webcam permissions when prompted.

### Production Build

Compile TypeScript and bundle optimized production assets into `dist/`:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

### Running Tests

Execute the DSP verification suite:

```bash
npm test
```

## Main Thread & Worker Pipeline

```
Main UI Thread:
[Camera] ──> [HTMLVideoElement] ──> [MediaPipe Tasks FaceLandmarker]
                                                │
                                    (Forehead & Cheek ROIs)
                                                │
                                       [extractMeanRGB]
                                                │
                                    postMessage({ type: 'frame', r, g, b, timestamp })
                                                ▼
Dedicated Web Worker (rppgWorker.ts):
┌─────────────────────────────────────────────────────────────────┐
│ 1. Circular RGB Signal Buffer                                   │
│ 2. Pulse Extraction (POS / CHROM / Green)                       │
│ 3. Zero-phase Bandpass Filter (0.7 - 3.5 Hz)                   │
│ 4. Welch Periodogram PSD & Peak Detection (BPM)                 │
│ 5. SNR Calculation & Inter-Beat Interval (IBI) HRV Estimation   │
└─────────────────────────────────────────────────────────────────┘
                                                │
                                    postMessage({ type: 'result', bpm, snr, pulseWave, ... })
                                                ▼
Main UI Thread:
[App.tsx] ──> Updates VitalCards, WaveformChart & SpectrumChart (60 FPS HUD)
```
