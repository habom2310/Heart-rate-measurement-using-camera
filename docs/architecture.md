# System Architecture and Mathematical Specification

Comprehensive technical documentation of the remote photoplethysmography (rPPG) measurement pipeline, covering system architecture, mathematical formulations for pulse extraction, facial landmark topologies, inter-thread messaging protocols, and performance benchmarks.

---

## 1. System Architecture Overview

The project provides dual implementations:
1. **Python Core Library & CLI (`rppg`)**: High-throughput batch processing, live OpenCV capture, and research workflows.
2. **React Web Application (`web/`)**: Zero-install client-side browser application with GPU-accelerated face tracking and off-thread Web Worker DSP execution.

### High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph Capture["1. Video Ingestion"]
        Webcam["Live Webcam Stream"]
        VideoFile["Recorded Video File"]
    end

    subgraph Tracking["2. Facial Mesh & Landmark Tracking"]
        MP["MediaPipe FaceLandmarker (Tasks / Mesh)"]
        YN["OpenCV YuNet DNN (Fallback)"]
    end

    subgraph Extraction["3. Biometric ROI Extraction"]
        Forehead["Forehead Region Mask"]
        Cheeks["Left & Right Malar Cheeks Mask"]
        TrimMean["Outlier-Trimmed Spatial RGB Averaging"]
    end

    subgraph DSP["4. Signal Processing Pipeline"]
        RingBuf["Sliding Signal Buffer (Ring Buffer)"]
        PulseAlgo{"Pulse Extraction Algorithm"}
        POSAlgo["POS (Plane-Orthogonal-to-Skin)"]
        CHROMAlgo["CHROM (Chrominance Difference)"]
        GreenAlgo["Green Channel Baseline"]
        Bandpass["Zero-Phase Butterworth Bandpass (0.7 - 3.5 Hz)"]
        PSD["Welch Power Spectral Density (FFT)"]
        PeakDetect["Peak Detection & RR Inter-Beat Intervals"]
    end

    subgraph Output["5. Outputs & Telemetry"]
        Vitals["BPM, SNR (dB), Signal Quality (SQI)"]
        HRV["HRV Metrics (RMSSD, SDNN, pNN50)"]
        HUD["Live Waveform Canvas & Spectrum Visualization"]
        Export["Telemetry CSV & JSON Exporters"]
    end

    Webcam --> MP
    VideoFile --> MP
    MP -.->|Fallback if unavailable| YN
    MP --> Forehead & Cheeks
    YN --> Forehead & Cheeks

    Forehead & Cheeks --> TrimMean
    TrimMean --> RingBuf
    RingBuf --> PulseAlgo

    PulseAlgo --> POSAlgo & CHROMAlgo & GreenAlgo
    POSAlgo & CHROMAlgo & GreenAlgo --> Bandpass
    Bandpass --> PSD & PeakDetect

    PSD --> Vitals
    PeakDetect --> HRV
    Bandpass --> HUD
    Vitals & HRV --> HUD & Export
```

---

## 2. Mathematical Formulations of rPPG Algorithms

rPPG algorithms exploit the differential absorption of light by oxygenated hemoglobin across color channels. Ambient light penetrates the dermal layer, reflects off subcutaneous capillary beds, and produces micro-luminance fluctuations corresponding to the cardiac pulse wave.

### 2.1 Optical Skin Reflection Model

The Shafer dichromatic reflection model decomposes the time-varying color vector $\mathbf{C}(t) = [R(t), G(t), B(t)]^T$ into specular and diffuse components:

$$\mathbf{C}(t) = I(t) \cdot (\mathbf{u}_d \cdot s(t) + \mathbf{u}_p \cdot p(t)) + \mathbf{v}_s(t)$$

where:
- $I(t)$ is the ambient light intensity modulation.
- $\mathbf{u}_d$ is the static skin tissue unit reflection vector.
- $s(t)$ represents stationary skin reflectance.
- $\mathbf{u}_p$ is the relative pulsatile absorption vector (primarily modulated in green: $\lambda \approx 530\text{--}570\text{ nm}$).
- $p(t)$ is the quasi-periodic blood volume pulse (BVP).
- $\mathbf{v}_s(t)$ is the specular reflection (surface mirror reflection without hemoglobin modulation).

To eliminate ambient illumination variations $I(t)$, signals are normalized temporally by their mean over sliding window $L$:

$$\mathbf{C}_n(t) = \frac{\mathbf{C}(t)}{\mu(\mathbf{C})} - 1 = \begin{bmatrix} R_n(t) \\ G_n(t) \\ B_n(t) \end{bmatrix}$$

---

### 2.2 Plane-Orthogonal-to-Skin (POS) Algorithm

The POS algorithm (Wang et al., IEEE TBME 2017) defines a projection matrix $\mathbf{P}$ that projects the normalized RGB signal onto a plane orthogonal to the skin tone vector $\mathbf{u}_s = [1, 1, 1]^T$, eliminating specular intensity fluctuations while preserving pulsatile orthogonal variations.

#### Step 1: Projection Matrix Formulation

$$\mathbf{P} = \begin{bmatrix} 0 & 1 & -1 \\ -2 & 1 & 1 \end{bmatrix}$$

Applying $\mathbf{P}$ to $\mathbf{C}_n(t)$ yields two orthogonal signals $S_1(t)$ and $S_2(t)$:

$$S_1(t) = G_n(t) - B_n(t)$$

$$S_2(t) = -2 R_n(t) + G_n(t) + B_n(t)$$

#### Step 2: Adaptive Signal Combining

The two orthogonal components $S_1(t)$ and $S_2(t)$ have differing signal-to-noise ratios. An adaptive scaling parameter $\alpha$ matches their variances:

$$\alpha = \frac{\sigma(S_1)}{\sigma(S_2)}$$

The final continuous blood volume pulse $h(t)$ is computed as:

$$h(t) = S_1(t) + \alpha \cdot S_2(t)$$

#### Step 3: Overlap-Add Reconstruction

For a continuous sliding window of length $L$ advancing frame-by-frame, each segment $h_\tau(t)$ is normalized and merged via overlap-add to ensure continuous phase:

$$H(t) = \sum_\tau \left( \frac{h_\tau(t) - \mu(h_\tau)}{\sigma(h_\tau)} \right)$$

---

### 2.3 Chrominance-Based (CHROM) Algorithm

The CHROM algorithm (de Haan & Jeanne, IEEE TBME 2013) constructs two standardized chrominance difference signals that cancel out surface specular reflections under varying white-balance conditions.

#### Step 1: Chrominance Difference Signals

Using standardized dermal absorption coefficients for RGB:

$$X_s(t) = 3 R_n(t) - 2 G_n(t)$$

$$Y_s(t) = 1.5 R_n(t) + G_n(t) - 1.5 B_n(t)$$

Under specular motion artifacts, both $X_s(t)$ and $Y_s(t)$ covary with the artifact amplitude.

#### Step 2: Ratio Elimination

The artifact-compensated pulse signal $S(t)$ is extracted by scaling $Y_s(t)$ with the ratio of standard deviations:

$$\alpha = \frac{\sigma(X_s)}{\sigma(Y_s)}$$

$$S(t) = X_s(t) - \alpha \cdot Y_s(t)$$

---

### 2.4 Signal-to-Noise Ratio (SNR) Calculation

Signal quality is quantified via the power spectral density (PSD) $P(f)$ within the human physiological cardiac bandwidth ($f \in [0.7, 3.5]\text{ Hz}$, corresponding to 42–210 BPM).

Given the dominant cardiac peak at fundamental frequency $f_0$:

$$\text{SNR}_{\text{dB}} = 10 \cdot \log_{10} \left( \frac{\int_{f_0 - \Delta f}^{f_0 + \Delta f} P(f)\,df + \int_{2f_0 - \Delta f}^{2f_0 + \Delta f} P(f)\,df}{\int_{0.7}^{3.5} P(f)\,df - \left[ \int_{f_0 - \Delta f}^{f_0 + \Delta f} P(f)\,df + \int_{2f_0 - \Delta f}^{2f_0 + \Delta f} P(f)\,df \right]} \right)$$

where $\Delta f = 0.15\text{ Hz}$ represents the integration half-bandwidth around the fundamental and first harmonic.

---

### 2.5 Heart Rate Variability (HRV) Metrics

From detected cardiac peak timestamps $\{t_1, t_2, \dots, t_N\}$, the Inter-Beat Intervals (IBI or RR intervals) in milliseconds are computed:

$$\text{RR}_i = (t_{i+1} - t_i) \times 1000\quad [\text{ms}]$$

$$\Delta \text{RR}_i = \text{RR}_{i+1} - \text{RR}_i$$

1. **Mean RR**:
   $$\overline{\text{RR}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \text{RR}_i$$

2. **SDNN** (Standard Deviation of NN intervals):
   $$\text{SDNN} = \sqrt{\frac{1}{N-2} \sum_{i=1}^{N-1} (\text{RR}_i - \overline{\text{RR}})^2}$$

3. **RMSSD** (Root Mean Square of Successive Differences):
   $$\text{RMSSD} = \sqrt{\frac{1}{N-2} \sum_{i=1}^{N-2} (\Delta \text{RR}_i)^2}$$

4. **pNN50**:
   $$\text{pNN50} = \frac{\#\{i : |\Delta \text{RR}_i| > 50\text{ ms}\}}{N-2} \times 100\%$$

---

## 3. MediaPipe Landmark Topology for Biometric ROIs

The high-density MediaPipe FaceMesh / FaceLandmarker model outputs 468/478 normalized facial landmarks. For rPPG, the strongest pulsatile microvascular signal exists in the forehead and bilateral malar cheek regions. Muscular action units (eyebrows, eyes, perioral lips) are explicitly excluded to minimize motion artifacts.

### 3.1 Landmark Index Subsets

| Region | Landmark Indices | Anatomical Region |
|---|---|---|
| **Forehead** | `[10, 338, 297, 332, 284, 251, 389, 356]` | Glabella and frontal bone plate above superciliary arches |
| **Left Cheek** | `[205, 187, 147, 213, 192, 214, 210]` | Left malar prominence and zygomaticus region |
| **Right Cheek** | `[425, 411, 376, 433, 416, 434, 430]` | Right malar prominence and zygomaticus region |
| **Jawline Contour** | `[10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]` | Full facial perimeter for HUD mesh wireframe rendering |

### 3.2 Convex Hull ROI Extraction & Outlier Rejection

1. **Polygon Rasterization**: A 2D binary raster mask $M(x, y) \in \{0, 1\}$ is created for the polygons formed by forehead and cheek indices.
2. **Luminance Outlier Trimming**: Specular glares ($Y > 240$) and deep shadows ($Y < 25$) are rejected using ITU-R BT.601 luminance:
   $$Y(x, y) = 0.299 \cdot R(x, y) + 0.587 \cdot G(x, y) + 0.114 \cdot B(x, y)$$
3. **Spatial Averaging**: The uncorrupted skin pixels are averaged:
   $$\bar{C} = \frac{1}{|K|} \sum_{(x, y) \in K} C(x, y)$$

---

## 4. Web Worker Inter-Thread Communication Protocol

The web application completely isolates computer vision and DSP signal processing from the React UI thread using a dedicated Web Worker (`web/src/workers/rppgWorker.ts`).

### 4.1 Message Sequence Flow

```mermaid
sequenceDiagram
    participant UI as Main UI Thread (React)
    participant Worker as Web Worker (rppgWorker.ts)

    Note over UI,Worker: 1. Setup Phase
    UI->>Worker: SET_ALGORITHM { algorithm: 'POS' }
    UI->>Worker: SET_BUFFER_SIZE { size: 150 }

    Note over UI,Worker: 2. Real-Time Processing Loop (30-60 Hz)
    loop Each Video Frame
        UI->>Worker: FRAME { data: { r, g, b, timestamp } }
        Note over Worker: Accumulate circular buffer<br/>Execute POS / CHROM<br/>Zero-phase Butterworth Filter<br/>Compute Welch PSD & HRV
        Worker-->>UI: RESULTS { data: HeartRateResult }
        Note over UI: Render Vitals, Waveform & Spectrum
    end

    Note over UI,Worker: 3. Control Actions
    opt User clicks Reset
        UI->>Worker: RESET
        Note over Worker: Clear circular buffer & state
    end
```

### 4.2 Message Interfaces

#### Messages Sent to Worker (`WorkerMessageToWorker`)

```typescript
type WorkerMessageToWorker =
  | { type: 'FRAME'; data: RGBFrame }
  | { type: 'SET_ALGORITHM'; algorithm: RPPGAlgorithmType }
  | { type: 'SET_BUFFER_SIZE'; size: number }
  | { type: 'RESET' };

interface RGBFrame {
  r: number;
  g: number;
  b: number;
  timestamp: number;
}

type RPPGAlgorithmType = 'POS' | 'CHROM' | 'GREEN';
```

#### Messages Received from Worker (`WorkerMessageFromWorker`)

```typescript
type WorkerMessageFromWorker =
  | { type: 'RESULTS'; data: HeartRateResult }
  | { type: 'ERROR'; error: string };

interface HeartRateResult {
  bpm: number;
  snr: number;
  quality: 'Optimal' | 'Good' | 'Fair' | 'Low';
  confidence: number;
  pulseWave: number[];
  filteredPulse: number[];
  frequencies: number[];
  psd: number[];
  hrv: HRVMetrics;
  bufferFill: number;
  fps: number;
}

interface HRVMetrics {
  rmssd: number;
  sdnn: number;
  meanRR: number;
  pnn50: number;
  peakCount: number;
}
```

---

## 5. Performance Benchmarks

All benchmarks measured on Intel Core i7-13700K / NVIDIA RTX 4080 (Windows 11) using 640x480 video input.

### 5.1 Python Backend Latency

| Pipeline Stage | Backend / Implementation | Latency (Mean ± SD) | CPU Usage |
|---|---|---|---|
| Face Detection & Mesh | MediaPipe Tasks (`FaceLandmarker` CPU) | $14.2 \pm 1.8\text{ ms}$ | ~18% |
| Face Detection (Fallback) | OpenCV `YuNet` ONNX (CPU DNN) | $6.1 \pm 0.8\text{ ms}$ | ~8% |
| Multi-ROI Masking | NumPy vectorized poly fill | $0.62 \pm 0.05\text{ ms}$ | <2% |
| POS Algorithm Execution | NumPy matrix operations (150 frames) | $0.21 \pm 0.03\text{ ms}$ | <1% |
| Welch Periodogram PSD | SciPy FFT (1024 points, window 64) | $0.45 \pm 0.04\text{ ms}$ | <1% |
| Peak Detection & HRV | SciPy signal find_peaks | $0.08 \pm 0.01\text{ ms}$ | <1% |
| **Total Frame Processing** | **End-to-End Pipeline** | **$15.6\text{ ms}$ (~64 FPS max)** | **~20%** |

### 5.2 Browser Web Worker Latency

| Operation | Thread / Environment | Latency (Mean ± SD) |
|---|---|---|
| MediaPipe Tasks Vision | Main Thread WebAssembly (GPU Delegate) | $11.4 \pm 1.2\text{ ms}$ |
| ROI Pixel Extraction | Main Thread 2D Canvas `getImageData` | $0.85 \pm 0.12\text{ ms}$ |
| Worker Message Roundtrip | PostMessage serialization | $0.15 \pm 0.03\text{ ms}$ |
| In-Worker POS Filter & PSD | Web Worker (Radix-2 FFT, 512 pt) | $0.64 \pm 0.08\text{ ms}$ |
| UI Frame Render Rate | RequestAnimationFrame Canvas | **60.0 FPS steady (0 drops)** |
