"""rppg: Real-time remote photoplethysmography heart rate measurement package."""

from rppg.core.algorithms import (
    CHROMAlgorithm,
    GreenMeanAlgorithm,
    POSAlgorithm,
    RPPGAlgorithm,
)
from rppg.core.hrv import (
    HRVMetrics,
    calculate_hrv_metrics,
    detect_peaks,
)
from rppg.core.roi import (
    FOREHEAD_INDICES,
    LEFT_CHEEK_INDICES,
    RIGHT_CHEEK_INDICES,
    draw_roi_overlay,
    extract_mean_rgb,
    extract_rois_mask,
)
from rppg.core.signal import (
    HeartRateResult,
    SignalBuffer,
    butter_bandpass,
    butter_bandpass_filter,
    compute_welch_psd,
    detrend_signal,
    estimate_heart_rate,
    interpolate_signal,
)
from rppg.core.tracker import (
    BaseTracker,
    FaceLandmarksResult,
    MediaPipeTracker,
    YuNetTracker,
)
from rppg.pipeline import (
    PipelineResult,
    RPPGPipeline,
)

__version__ = "0.2.0"

__all__ = [
    "BaseTracker",
    "CHROMAlgorithm",
    "FaceLandmarksResult",
    "FOREHEAD_INDICES",
    "GreenMeanAlgorithm",
    "HRVMetrics",
    "HeartRateResult",
    "LEFT_CHEEK_INDICES",
    "MediaPipeTracker",
    "POSAlgorithm",
    "PipelineResult",
    "RIGHT_CHEEK_INDICES",
    "RPPGAlgorithm",
    "RPPGPipeline",
    "SignalBuffer",
    "YuNetTracker",
    "__version__",
    "butter_bandpass",
    "butter_bandpass_filter",
    "calculate_hrv_metrics",
    "compute_welch_psd",
    "detect_peaks",
    "detrend_signal",
    "draw_roi_overlay",
    "estimate_heart_rate",
    "extract_mean_rgb",
    "extract_rois_mask",
    "interpolate_signal",
]
