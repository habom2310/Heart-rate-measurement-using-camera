"""Unit tests for the end-to-end rPPG processing pipeline and CLI."""

from __future__ import annotations

import numpy as np
import pytest

from rppg.cli import create_algorithm, create_tracker, draw_hud, parse_args
from rppg.core.algorithms import CHROMAlgorithm, GreenMeanAlgorithm, POSAlgorithm
from rppg.core.roi import FOREHEAD_INDICES, LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES
from rppg.core.tracker import BaseTracker, FaceLandmarksResult
from rppg.pipeline import PipelineResult, RPPGPipeline


class MockFaceTracker(BaseTracker):
    """Mock face tracker returning synthetic face mesh coordinates."""

    def __init__(self, detected: bool = True) -> None:
        self.detected = detected

    def process(self, frame_bgr: np.ndarray) -> FaceLandmarksResult | None:
        if not self.detected or frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]
        landmarks = np.zeros((468, 2), dtype=np.float32)

        # Populate forehead and cheeks
        for i, idx in enumerate(FOREHEAD_INDICES):
            landmarks[idx] = [w * 0.4 + (i * 2), h * 0.2]
        for i, idx in enumerate(LEFT_CHEEK_INDICES):
            landmarks[idx] = [w * 0.3 + (i * 2), h * 0.5]
        for i, idx in enumerate(RIGHT_CHEEK_INDICES):
            landmarks[idx] = [w * 0.6 + (i * 2), h * 0.5]

        norm_lm = landmarks.copy()
        norm_lm[:, 0] /= max(w, 1)
        norm_lm[:, 1] /= max(h, 1)

        return FaceLandmarksResult(
            bbox=(int(w * 0.2), int(h * 0.1), int(w * 0.6), int(h * 0.8)),
            landmarks=landmarks,
            normalized_landmarks=norm_lm,
            confidence=0.99,
        )


def test_pipeline_initialization():
    """Verify RPPGPipeline initialization and parameter constraints."""
    pipeline = RPPGPipeline(
        tracker=MockFaceTracker(),
        algorithm=POSAlgorithm(),
        buffer_size=120,
        min_samples=45,
    )
    assert pipeline.buffer_size == 120
    assert pipeline.min_samples == 45
    assert len(pipeline._signal_buffer) == 0

    with pytest.raises(ValueError):
        RPPGPipeline(buffer_size=-1)

    with pytest.raises(ValueError):
        RPPGPipeline(buffer_size=50, min_samples=60)


def test_pipeline_no_face_detected():
    """Verify pipeline output when no face is found in input frame."""
    tracker = MockFaceTracker(detected=False)
    pipeline = RPPGPipeline(tracker=tracker, min_samples=10)

    frame = np.full((100, 100, 3), 120, dtype=np.uint8)
    res = pipeline.process_frame(frame, timestamp=0.0)

    assert isinstance(res, PipelineResult)
    assert not res.face_detected
    assert res.bpm is None
    assert res.snr is None
    assert res.raw_rgb == (0.0, 0.0, 0.0)


def test_pipeline_synthetic_pulsatile_signal():
    """Verify end-to-end processing with synthetic pulsatile face frames."""
    fps = 30.0
    freq_hz = 1.2  # 1.2 Hz = 72 BPM
    n_frames = 90  # 3 seconds of video

    tracker = MockFaceTracker(detected=True)
    pipeline = RPPGPipeline(
        tracker=tracker,
        algorithm=POSAlgorithm(),
        buffer_size=150,
        min_samples=60,
    )

    t_arr = np.arange(n_frames) / fps
    results: list[PipelineResult] = []

    # Nominal skin base color (BGR): B=100, G=140, R=190
    for i, t in enumerate(t_arr):
        pulse = 4.0 * np.sin(2.0 * np.pi * freq_hz * t)
        # Pulse is strongest in Green channel and slightly in Red
        b = int(np.clip(100.0 - 0.5 * pulse, 0, 255))
        g = int(np.clip(140.0 + 1.0 * pulse, 0, 255))
        r = int(np.clip(190.0 + 0.3 * pulse, 0, 255))

        frame = np.full((120, 120, 3), (b, g, r), dtype=np.uint8)
        res = pipeline.process_frame(frame, timestamp=t)
        results.append(res)

    # First 59 frames should not yet have BPM (under min_samples=60)
    assert results[50].bpm is None
    assert results[50].face_detected is True
    assert len(results[50].pulse_wave) == 0

    # Frame 60 onwards should produce valid estimates
    final_res = results[-1]
    assert final_res.face_detected is True
    assert final_res.bpm is not None
    # Estimated heart rate should be close to 72 BPM (within 5 BPM)
    assert pytest.approx(final_res.bpm, abs=5.0) == 72.0
    assert final_res.snr is not None
    assert len(final_res.pulse_wave) == n_frames
    assert len(final_res.freqs) > 0
    assert len(final_res.psd) > 0


def test_pipeline_reset():
    """Verify that reset clears buffers and timestamps."""
    pipeline = RPPGPipeline(tracker=MockFaceTracker(), min_samples=10)
    frame = np.full((100, 100, 3), 100, dtype=np.uint8)

    for i in range(15):
        pipeline.process_frame(frame, timestamp=i * (1.0 / 30.0))

    assert len(pipeline._signal_buffer) == 15
    pipeline.reset()
    assert len(pipeline._signal_buffer) == 0
    assert len(pipeline._timestamps) == 0


def test_cli_argument_parsing():
    """Verify CLI argument parser defaults and flags."""
    args = parse_args(["--source", "test_video.mp4", "--algo", "chrom", "--buffer-size", "200"])
    assert args.source == "test_video.mp4"
    assert args.algo == "chrom"
    assert args.buffer_size == 200
    assert not args.no_display


def test_cli_factory_helpers():
    """Verify algorithm and tracker instantiation factories."""
    assert isinstance(create_algorithm("pos"), POSAlgorithm)
    assert isinstance(create_algorithm("chrom"), CHROMAlgorithm)
    assert isinstance(create_algorithm("green"), GreenMeanAlgorithm)

    with pytest.raises(ValueError):
        create_algorithm("unknown_algo")

    with pytest.raises(ValueError):
        create_tracker("unknown_tracker")


def test_cli_draw_hud():
    """Verify HUD drawing on synthetic frame."""
    frame = np.full((240, 320, 3), 50, dtype=np.uint8)
    dummy_res = PipelineResult(
        frame=frame,
        raw_rgb=(180.0, 140.0, 100.0),
        bpm=75.0,
        snr=8.5,
        hrv=None,
        pulse_wave=np.sin(np.linspace(0, 10, 100)),
        freqs=np.linspace(0.5, 4.0, 50),
        psd=np.ones(50),
        fps=30.0,
        face_detected=True,
    )

    annotated = draw_hud(frame.copy(), dummy_res)
    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)
