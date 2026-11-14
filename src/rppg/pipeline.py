"""Pipeline orchestrator for end-to-end rPPG processing.

Integrates face tracking, ROI extraction, spatial RGB signal accumulation,
pulse extraction algorithms, peak detection, and HRV metric calculation into a
clean, decoupled processing pipeline.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from rppg.core.algorithms import POSAlgorithm, RPPGAlgorithm
from rppg.core.hrv import HRVMetrics, calculate_hrv_metrics, detect_peaks
from rppg.core.roi import draw_roi_overlay, extract_mean_rgb, extract_rois_mask
from rppg.core.signal import SignalBuffer, compute_welch_psd, estimate_heart_rate
from rppg.core.tracker import BaseTracker, MediaPipeTracker, YuNetTracker


@dataclass
class PipelineResult:
    """Outputs produced by RPPGPipeline for a single processed frame.

    Attributes:
        frame: Output frame with HUD tracking annotations.
        raw_rgb: Spatial mean (R, G, B) values extracted from the face ROIs.
        bpm: Estimated heart rate in beats per minute, or None if insufficient data.
        snr: Signal-to-noise ratio in decibels (dB), or None if insufficient data.
        hrv: Heart rate variability metrics, or None if insufficient peaks detected.
        pulse_wave: Extracted 1D blood volume pulse signal array.
        freqs: Frequency bins (Hz) corresponding to the power spectral density.
        psd: Power spectral density values of the pulse signal.
        fps: Effective instantaneous or rolling frame rate in Hz.
        face_detected: True if a face was detected and tracked in the frame.
    """

    frame: np.ndarray
    raw_rgb: tuple[float, float, float]
    bpm: Optional[float]
    snr: Optional[float]
    hrv: Optional[HRVMetrics]
    pulse_wave: np.ndarray
    freqs: np.ndarray
    psd: np.ndarray
    fps: float
    face_detected: bool


class RPPGPipeline:
    """Decoupled orchestrator for real-time camera or video rPPG estimation."""

    def __init__(
        self,
        tracker: Optional[BaseTracker] = None,
        algorithm: Optional[RPPGAlgorithm] = None,
        buffer_size: int = 150,
        min_samples: int = 60,
    ) -> None:
        """Initialize the rPPG processing pipeline.

        Args:
            tracker: Face tracking detector. Defaults to MediaPipeTracker with
                automatic fallback to YuNetTracker.
            algorithm: Core rPPG algorithm. Defaults to POSAlgorithm().
            buffer_size: Ring buffer capacity in frames (default: 150, ~5 s at 30 FPS).
            min_samples: Minimum required frames before calculating metrics (default: 60, ~2 s).
        """
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        if min_samples <= 0:
            raise ValueError(f"min_samples must be positive, got {min_samples}")
        if min_samples > buffer_size:
            raise ValueError(
                f"min_samples ({min_samples}) cannot exceed buffer_size ({buffer_size})"
            )

        if tracker is None:
            try:
                self.tracker: BaseTracker = MediaPipeTracker()
            except Exception:
                self.tracker = YuNetTracker()
        else:
            self.tracker = tracker

        self.algorithm: RPPGAlgorithm = algorithm or POSAlgorithm()
        self.buffer_size = buffer_size
        self.min_samples = min_samples

        self._signal_buffer = SignalBuffer(capacity=buffer_size)
        self._timestamps: collections.deque[float] = collections.deque(maxlen=30)
        self._fps: float = 30.0

    @property
    def fps(self) -> float:
        """Current estimated rolling frame rate in Hz."""
        return self._fps

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> PipelineResult:
        """Process a single video frame and return real-time rPPG estimates.

        Args:
            frame_bgr: Video frame in BGR format of shape (H, W, 3).
            timestamp: Frame presentation timestamp in seconds. If None,
                system time (`time.time()`) is used.

        Returns:
            PipelineResult: Dataclass containing annotated frame, RGB signals,
                estimated BPM, SNR, HRV, pulse waveform, and PSD.
        """
        t = time.time() if timestamp is None else float(timestamp)
        self._update_fps(t)

        if frame_bgr is None or frame_bgr.size == 0:
            return self._empty_result(frame_bgr, face_detected=False)

        tracker_result = self.tracker.process(frame_bgr)

        if tracker_result is not None:
            face_detected = True
            mask = extract_rois_mask(frame_bgr.shape, tracker_result.landmarks)
            raw_rgb = extract_mean_rgb(frame_bgr, mask)

            # Append to color buffer with timestamp
            self._signal_buffer.append(raw_rgb, timestamp=t)
            annotated_frame = draw_roi_overlay(frame_bgr, tracker_result.landmarks)
        else:
            face_detected = False
            raw_rgb = (0.0, 0.0, 0.0)
            annotated_frame = frame_bgr.copy()

        # Compute pulse signal and metrics if sufficient samples are available
        if len(self._signal_buffer) >= self.min_samples:
            rgb_data = self._signal_buffer.get_rgb()
            effective_fps = max(5.0, min(self._fps, 120.0))

            pulse_wave = self.algorithm.process(rgb_data, fps=effective_fps)
            hr_result = estimate_heart_rate(pulse_wave, fs=effective_fps)

            bpm = float(hr_result.bpm) if hr_result.bpm > 0 else None
            snr = float(hr_result.snr)

            peaks = detect_peaks(pulse_wave, fs=effective_fps)
            hrv = calculate_hrv_metrics(peaks, fs=effective_fps) if len(peaks) >= 2 else None
            freqs, psd = compute_welch_psd(pulse_wave, fs=effective_fps)
        else:
            bpm = None
            snr = None
            hrv = None
            pulse_wave = np.empty(0, dtype=np.float64)
            freqs = np.empty(0, dtype=np.float64)
            psd = np.empty(0, dtype=np.float64)

        return PipelineResult(
            frame=annotated_frame,
            raw_rgb=raw_rgb,
            bpm=bpm,
            snr=snr,
            hrv=hrv,
            pulse_wave=pulse_wave,
            freqs=freqs,
            psd=psd,
            fps=self._fps,
            face_detected=face_detected,
        )

    def reset(self) -> None:
        """Clear all stored buffer samples and reset frame rate estimation."""
        self._signal_buffer.clear()
        self._timestamps.clear()
        self._fps = 30.0

    def close(self) -> None:
        """Release tracker and any underlying resources."""
        if hasattr(self.tracker, "close"):
            self.tracker.close()

    def __enter__(self) -> RPPGPipeline:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _update_fps(self, current_time: float) -> None:
        """Update rolling FPS estimate based on inter-frame timestamp deltas."""
        self._timestamps.append(current_time)
        if len(self._timestamps) >= 2:
            dt = self._timestamps[-1] - self._timestamps[0]
            count = len(self._timestamps) - 1
            if dt > 1e-4:
                instantaneous_fps = count / dt
                # Clamp within sensible camera frame rate bounds
                self._fps = float(np.clip(instantaneous_fps, 5.0, 120.0))

    def _empty_result(self, frame: np.ndarray, face_detected: bool) -> PipelineResult:
        """Helper to construct an empty/failed frame result."""
        empty_arr = np.empty(0, dtype=np.float64)
        return PipelineResult(
            frame=frame if frame is not None else np.zeros((1, 1, 3), dtype=np.uint8),
            raw_rgb=(0.0, 0.0, 0.0),
            bpm=None,
            snr=None,
            hrv=None,
            pulse_wave=empty_arr,
            freqs=empty_arr,
            psd=empty_arr,
            fps=self._fps,
            face_detected=face_detected,
        )
