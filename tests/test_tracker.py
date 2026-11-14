"""Unit tests for face trackers and ROI extraction."""

from __future__ import annotations

import numpy as np
import pytest

from rppg.core.roi import (
    FOREHEAD_INDICES,
    LEFT_CHEEK_INDICES,
    RIGHT_CHEEK_INDICES,
    draw_roi_overlay,
    extract_mean_rgb,
    extract_rois_mask,
)
from rppg.core.tracker import (
    BaseTracker,
    FaceLandmarksResult,
    MediaPipeTracker,
    YuNetTracker,
)


def test_face_landmarks_result_dataclass():
    """Verify FaceLandmarksResult creation and attributes."""
    bbox = (10, 20, 100, 120)
    lm = np.zeros((468, 2), dtype=np.float32)
    norm_lm = np.zeros((468, 2), dtype=np.float32)
    conf = 0.95

    res = FaceLandmarksResult(
        bbox=bbox,
        landmarks=lm,
        normalized_landmarks=norm_lm,
        confidence=conf,
    )
    assert res.bbox == bbox
    assert res.landmarks.shape == (468, 2)
    assert res.normalized_landmarks.shape == (468, 2)
    assert res.confidence == 0.95


def test_extract_rois_mask_dense_mesh():
    """Verify mask extraction with full MediaPipe mesh indices."""
    shape = (200, 200, 3)
    landmarks = np.zeros((468, 2), dtype=np.float32)

    # Set valid coordinates for forehead and cheek indices
    for i, idx in enumerate(FOREHEAD_INDICES):
        landmarks[idx] = [50 + (i * 10), 30]
    for i, idx in enumerate(LEFT_CHEEK_INDICES):
        landmarks[idx] = [40 + (i * 10), 100]
    for i, idx in enumerate(RIGHT_CHEEK_INDICES):
        landmarks[idx] = [120 + (i * 10), 100]

    mask = extract_rois_mask(shape, landmarks)
    assert mask.shape == (200, 200)
    assert mask.dtype == np.uint8
    # All values must be 0 or 255
    assert set(np.unique(mask)).issubset({0, 255})
    assert np.count_nonzero(mask) > 0


def test_extract_rois_mask_sparse_landmarks():
    """Verify fallback mask extraction for 5-point sparse landmarks (YuNet)."""
    shape = (200, 200)
    # 5 landmarks: right eye, left eye, nose, right mouth, left mouth
    landmarks = np.array(
        [
            [60.0, 70.0],   # right eye
            [120.0, 70.0],  # left eye
            [90.0, 100.0],  # nose
            [70.0, 130.0],  # right mouth
            [110.0, 130.0], # left mouth
        ],
        dtype=np.float32,
    )

    mask = extract_rois_mask(shape, landmarks)
    assert mask.shape == (200, 200)
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) > 0


def test_extract_rois_mask_empty():
    """Verify mask extraction when landmarks are empty or None."""
    shape = (100, 100)
    mask = extract_rois_mask(shape, np.empty((0, 2), dtype=np.float32))
    assert mask.shape == (100, 100)
    assert np.count_nonzero(mask) == 0


def test_extract_mean_rgb_synthetic_frame():
    """Verify RGB spatial mean extraction and BGR -> RGB conversion."""
    h, w = 100, 100
    # Create uniform BGR image: B=20, G=80, R=140
    frame_bgr = np.full((h, w, 3), (20, 80, 140), dtype=np.uint8)

    # Circular/square mask
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[20:80, 20:80] = 255

    r, g, b = extract_mean_rgb(frame_bgr, mask, trim_outliers=0.0)
    assert pytest.approx(r, abs=1e-3) == 140.0
    assert pytest.approx(g, abs=1e-3) == 80.0
    assert pytest.approx(b, abs=1e-3) == 20.0


def test_extract_mean_rgb_outlier_trimming():
    """Verify outlier trimming rejects extreme specular highlights."""
    h, w = 50, 50
    # Majority skin tone: B=30, G=70, R=120
    frame_bgr = np.full((h, w, 3), (30, 70, 120), dtype=np.uint8)

    mask = np.ones((h, w), dtype=np.uint8) * 255

    # Introduce extreme specular reflection in 2% of pixels: B=255, G=255, R=255
    frame_bgr[:2, :25] = (255, 255, 255)

    r_trimmed, g_trimmed, b_trimmed = extract_mean_rgb(frame_bgr, mask, trim_outliers=0.05)
    # The trimmed mean should be extremely close to the nominal 120, 70, 30
    assert pytest.approx(r_trimmed, abs=1.0) == 120.0
    assert pytest.approx(g_trimmed, abs=1.0) == 70.0
    assert pytest.approx(b_trimmed, abs=1.0) == 30.0


def test_extract_mean_rgb_empty_mask():
    """Verify extract_mean_rgb returns (0, 0, 0) when mask has no active pixels."""
    frame_bgr = np.full((50, 50, 3), 100, dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)

    rgb = extract_mean_rgb(frame_bgr, mask)
    assert rgb == (0.0, 0.0, 0.0)


def test_draw_roi_overlay():
    """Verify translucent HUD ROI overlay drawing."""
    h, w = 120, 120
    frame_bgr = np.full((h, w, 3), 50, dtype=np.uint8)

    landmarks = np.zeros((468, 2), dtype=np.float32)
    for i, idx in enumerate(FOREHEAD_INDICES):
        landmarks[idx] = [20 + (i * 10), 20]

    overlay = draw_roi_overlay(frame_bgr, landmarks, color=(0, 255, 136), alpha=0.3)
    assert overlay.shape == frame_bgr.shape
    assert overlay.dtype == frame_bgr.dtype
    # The overlay should not be identical to original frame where ROI is drawn
    assert not np.array_equal(overlay, frame_bgr)


def test_base_tracker_interface():
    """Verify BaseTracker abstract interface subclassing."""

    class DummyTracker(BaseTracker):
        def process(self, frame_bgr):
            return None

    tracker = DummyTracker()
    assert tracker.process(np.zeros((10, 10, 3), dtype=np.uint8)) is None
    # Context manager support
    with tracker as t:
        assert t is tracker


def test_mediapipe_tracker_lifecycle():
    """Verify MediaPipeTracker initialization and processing."""
    tracker = MediaPipeTracker(max_num_faces=1)

    # Processing black frame should return None (no face detected)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    result = tracker.process(frame)
    assert result is None

    # Processing empty / None frame
    assert tracker.process(np.empty((0, 0, 3), dtype=np.uint8)) is None
    assert tracker.process(None) is None
    tracker.close()


def test_yunet_tracker_lifecycle():
    """Verify YuNetTracker initialization and empty frame handling."""
    tracker = YuNetTracker()

    # Processing empty / black frame
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    result = tracker.process(frame)
    assert result is None
    assert tracker.process(None) is None
    tracker.close()
