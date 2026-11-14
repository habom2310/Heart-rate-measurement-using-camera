"""Region of Interest (ROI) extraction and visualization for rPPG.

Defines standard facial landmarks for forehead and cheeks (MediaPipe FaceMesh),
generates binary masks, calculates spatial mean RGB with outlier rejection,
and draws modern translucent HUD overlays.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

# Standard landmark indices for MediaPipe Face Mesh (468/478 points)
# Symmetrical Mid-Forehead (balanced across facial midline, avoiding hair & eyebrows)
FOREHEAD_INDICES: list[int] = [67, 109, 10, 338, 297, 299, 296, 336, 9, 107, 66, 69]
LEFT_CHEEK_INDICES: list[int] = [346, 347, 348, 329, 330, 280, 425, 266, 352]
RIGHT_CHEEK_INDICES: list[int] = [117, 118, 119, 100, 101, 50, 205, 36, 123]


def _get_roi_polygons(
    landmarks: np.ndarray,
    indices_list: Sequence[Sequence[int]] | None = None,
) -> list[np.ndarray]:
    """Helper to convert landmark points into ROI polygon vertex arrays."""
    if landmarks is None or len(landmarks) == 0:
        return []

    if indices_list is None:
        indices_list = [FOREHEAD_INDICES, LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES]

    # Check if landmarks have enough points to satisfy indices
    max_req_idx = max((max(idx) for idx in indices_list if idx), default=-1)
    if len(landmarks) > max_req_idx and max_req_idx >= 0:
        polys = []
        for idx in indices_list:
            if idx:
                pts = landmarks[list(idx)].astype(np.int32)
                hull = cv2.convexHull(pts)
                polys.append(hull.squeeze())
        return polys

    # Fallback for sparse landmarks (e.g. 5 points from YuNet)
    # 0: right eye, 1: left eye, 2: nose, 3: right mouth, 4: left mouth
    if len(landmarks) >= 5:
        re = landmarks[0]
        le = landmarks[1]
        rm = landmarks[3]
        lm = landmarks[4]

        eye_d = le - re
        dist = float(np.hypot(eye_d[0], eye_d[1]))
        if dist < 1.0:
            return []

        u = eye_d / dist
        # Perpendicular vector pointing upward (negative y in image space)
        v = np.array([u[1], -u[0]], dtype=np.float32)

        # 1. Forehead polygon
        eye_mid = (re + le) / 2.0
        fh_center = eye_mid + (0.55 * dist) * v
        hw, hh = 0.4 * dist, 0.18 * dist
        fh_pts = np.array(
            [
                fh_center - hw * u - hh * v,
                fh_center + hw * u - hh * v,
                fh_center + hw * u + hh * v,
                fh_center - hw * u + hh * v,
            ],
            dtype=np.int32,
        )

        # 2. Right cheek polygon (viewer's left)
        rc_center = (re + rm) / 2.0 - (0.15 * dist) * u
        rw, rh = 0.18 * dist, 0.18 * dist
        rc_pts = np.array(
            [
                rc_center - rw * u - rh * v,
                rc_center + rw * u - rh * v,
                rc_center + rw * u + rh * v,
                rc_center - rw * u + rh * v,
            ],
            dtype=np.int32,
        )

        # 3. Left cheek polygon (viewer's right)
        lc_center = (le + lm) / 2.0 + (0.15 * dist) * u
        lw, lh = 0.18 * dist, 0.18 * dist
        lc_pts = np.array(
            [
                lc_center - lw * u - lh * v,
                lc_center + lw * u - lh * v,
                lc_center + lw * u + lh * v,
                lc_center - lw * u + lh * v,
            ],
            dtype=np.int32,
        )

        return [fh_pts, rc_pts, lc_pts]

    return []


def extract_rois_mask(
    frame_shape: tuple[int, ...],
    landmarks: np.ndarray,
    indices_list: Sequence[Sequence[int]] | None = None,
) -> np.ndarray:
    """Create a binary mask combining forehead and cheek ROIs.

    Args:
        frame_shape: Image shape (H, W) or (H, W, C).
        landmarks: Facial landmarks array of shape (N, 2) in pixel coordinates.
        indices_list: Sequence of index lists defining facial polygons. Defaults to
            [FOREHEAD_INDICES, LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES].

    Returns:
        np.ndarray: Binary mask of shape (H, W) with uint8 values (255 for ROI, 0 elsewhere).
    """
    h, w = frame_shape[0], frame_shape[1]
    mask = np.zeros((h, w), dtype=np.uint8)

    polys = _get_roi_polygons(landmarks, indices_list=indices_list)
    if polys:
        cv2.fillPoly(mask, polys, 255)

    return mask


def extract_mean_rgb(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    trim_outliers: float = 0.05,
) -> tuple[float, float, float]:
    """Compute spatial average RGB across the masked region.

    Converts BGR input to RGB and optionally rejects specular highlights and
    deep shadows via luminance-based percentile trimming.

    Args:
        frame_bgr: Input image in BGR format of shape (H, W, 3).
        mask: Binary mask of shape (H, W) where pixels > 0 belong to ROIs.
        trim_outliers: Fraction of lowest and highest luminance pixels to reject (0.0 to < 0.5).

    Returns:
        tuple[float, float, float]: Average (R, G, B) channel values as floats.
    """
    if frame_bgr is None or mask is None or frame_bgr.size == 0 or mask.size == 0:
        return (0.0, 0.0, 0.0)

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    mask_bool = mask > 0
    if not np.any(mask_bool):
        return (0.0, 0.0, 0.0)

    # Extract pixels inside mask in BGR order
    pixels_bgr = frame_bgr[mask_bool]  # shape (K, 3)
    # Convert to RGB order
    pixels_rgb = pixels_bgr[:, [2, 1, 0]].astype(np.float64)

    # Trim specular reflection highlights and deep shadows
    if 0.0 < trim_outliers < 0.5 and len(pixels_rgb) >= 10:
        # Standard ITU-R BT.601 luminance
        lum = 0.299 * pixels_rgb[:, 0] + 0.587 * pixels_rgb[:, 1] + 0.114 * pixels_rgb[:, 2]
        p_low = np.percentile(lum, trim_outliers * 100.0)
        p_high = np.percentile(lum, (1.0 - trim_outliers) * 100.0)
        keep = (lum >= p_low) & (lum <= p_high)
        if np.any(keep):
            pixels_rgb = pixels_rgb[keep]

    mean_rgb = np.mean(pixels_rgb, axis=0)
    return (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]))


def draw_roi_overlay(
    frame_bgr: np.ndarray,
    landmarks: np.ndarray,
    indices_list: Sequence[Sequence[int]] | None = None,
    color: tuple[int, int, int] = (0, 255, 136),
    alpha: float = 0.3,
) -> np.ndarray:
    """Draw modern translucent HUD polygonal overlays around the tracking regions.

    Args:
        frame_bgr: Video frame in BGR format.
        landmarks: Facial landmarks array of shape (N, 2) in pixel space.
        indices_list: Sequence of index lists defining facial polygons. Defaults to
            [FOREHEAD_INDICES, LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES].
        color: Overlay color in BGR format (default: (0, 255, 136) - vibrant cyan-green).
        alpha: Transparency level for the filled polygon mask in [0.0, 1.0].

    Returns:
        np.ndarray: Modified copy of frame_bgr with translucent overlay and crisp borders.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    out = frame_bgr.copy()
    polys = _get_roi_polygons(landmarks, indices_list=indices_list)
    if not polys:
        return out

    # Draw filled polygons on translucent overlay
    overlay = out.copy()
    cv2.fillPoly(overlay, polys, color)
    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0.0, dst=out)

    # Draw sharp boundary lines with full intensity
    for poly in polys:
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    return out
