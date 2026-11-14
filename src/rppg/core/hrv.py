"""Heart Rate Variability (HRV) metrics and peak detection for rPPG signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.signal


@dataclass
class HRVMetrics:
    """Heart Rate Variability (HRV) time-domain metrics.

    Attributes:
        ibis: Array of inter-beat intervals (RR intervals) in milliseconds.
        mean_rr: Mean RR interval in milliseconds.
        sdnn: Standard deviation of NN intervals in milliseconds.
        rmssd: Root mean square of successive differences between adjacent RR intervals (ms).
        pnn50: Percentage of adjacent RR interval pairs differing by > 50 ms (%).
        mean_hr: Mean heart rate in beats per minute (BPM).
    """

    ibis: np.ndarray
    mean_rr: float
    sdnn: float
    rmssd: float
    pnn50: float
    mean_hr: float = 0.0


def detect_peaks(
    pulse_signal: np.ndarray | Sequence[float],
    fs: float = 30.0,
    min_distance: int | None = None,
) -> np.ndarray:
    """Detect cardiac pulse peaks using physiological constraints (45 - 210 BPM).

    Args:
        pulse_signal: 1D pulse signal (typically bandpass-filtered).
        fs: Sampling rate in Hz (default: 30.0).
        min_distance: Minimum number of samples between successive peaks.
            If None, derived from max physiological heart rate (210 BPM).

    Returns:
        np.ndarray: Indices of detected peaks.
    """
    arr = np.asarray(pulse_signal, dtype=float)
    if len(arr) < 3:
        return np.empty(0, dtype=int)

    if min_distance is None:
        # 210 BPM -> min interval = 60 / 210 = 0.2857 s
        min_distance = max(1, int(fs * 60.0 / 210.0))

    pulse_std = float(np.std(arr))
    prominence = 0.15 * pulse_std if pulse_std > 1e-8 else None

    peaks, _ = scipy.signal.find_peaks(
        arr,
        distance=min_distance,
        prominence=prominence,
    )
    return peaks


def calculate_hrv_metrics(
    peak_indices: np.ndarray | Sequence[int],
    fs: float = 30.0,
) -> HRVMetrics:
    """Calculate time-domain Heart Rate Variability (HRV) metrics from pulse peaks.

    Computes:
        - IBIs: Inter-beat intervals in milliseconds
        - Mean RR: Average IBI in ms
        - SDNN: Standard deviation of normal-to-normal intervals in ms
        - RMSSD: Root mean square of successive differences in ms
        - pNN50: Percentage of successive intervals differing by > 50 ms (%)

    Args:
        peak_indices: 1D array-like of peak sample indices.
        fs: Sampling rate in Hz (default: 30.0).

    Returns:
        HRVMetrics: Dataclass holding time-domain HRV metrics.
    """
    peaks = np.asarray(peak_indices, dtype=float)

    if len(peaks) < 2 or fs <= 0:
        return HRVMetrics(
            ibis=np.empty(0, dtype=float),
            mean_rr=0.0,
            sdnn=0.0,
            rmssd=0.0,
            pnn50=0.0,
            mean_hr=0.0,
        )

    # Inter-beat intervals in milliseconds
    ibis = (np.diff(peaks) / fs) * 1000.0

    mean_rr = float(np.mean(ibis))
    sdnn = float(np.std(ibis, ddof=1)) if len(ibis) > 1 else 0.0

    if len(ibis) >= 2:
        diff_rr = np.diff(ibis)
        rmssd = float(np.sqrt(np.mean(diff_rr**2)))
        nn50_count = int(np.sum(np.abs(diff_rr) > 50.0))
        pnn50 = float((nn50_count / len(diff_rr)) * 100.0)
    else:
        rmssd = 0.0
        pnn50 = 0.0

    mean_hr = float(60000.0 / mean_rr) if mean_rr > 0 else 0.0

    return HRVMetrics(
        ibis=np.round(ibis, 2),
        mean_rr=round(mean_rr, 2),
        sdnn=round(sdnn, 2),
        rmssd=round(rmssd, 2),
        pnn50=round(pnn50, 2),
        mean_hr=round(mean_hr, 2),
    )
