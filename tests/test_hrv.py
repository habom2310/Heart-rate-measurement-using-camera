"""Unit tests for HRV metric calculations and cardiac peak detection."""

import numpy as np
import pytest

from rppg.core.hrv import (
    HRVMetrics,
    calculate_hrv_metrics,
    detect_peaks,
)


def test_detect_peaks_synthetic_sine():
    """Verify detect_peaks accurately identifies peaks on a periodic pulse wave."""
    fs = 30.0
    duration = 5.0
    t = np.arange(0, duration, 1.0 / fs)
    freq = 1.25  # 75 BPM, period = 0.8s = 24 samples at 30 Hz

    signal = np.sin(2 * np.pi * freq * t)
    peaks = detect_peaks(signal, fs=fs)

    assert len(peaks) >= 5

    # Check intervals between peaks are ~24 samples (0.8s)
    intervals = np.diff(peaks)
    assert np.all(intervals >= int(fs * 60.0 / 210.0))  # within physiological distance
    assert np.allclose(intervals, 24, atol=1)


def test_calculate_hrv_metrics_regular():
    """Verify HRV metrics for perfectly regular heart beats (zero variability)."""
    fs = 30.0
    # Beat every 24 samples = 0.8 s = 800 ms (75 BPM)
    peak_indices = np.array([24, 48, 72, 96, 120, 144, 168, 192])

    metrics = calculate_hrv_metrics(peak_indices, fs=fs)

    assert isinstance(metrics, HRVMetrics)
    assert len(metrics.ibis) == len(peak_indices) - 1
    assert np.isclose(metrics.mean_rr, 800.0, atol=1.0)
    assert np.isclose(metrics.sdnn, 0.0, atol=1e-2)
    assert np.isclose(metrics.rmssd, 0.0, atol=1e-2)
    assert metrics.pnn50 == 0.0
    assert np.isclose(metrics.mean_hr, 75.0, atol=0.1)


def test_calculate_hrv_metrics_variable():
    """Verify HRV metrics for known alternating RR intervals."""
    fs = 1000.0  # 1 sample = 1 ms for exact mathematical verification
    # Alternate intervals between 750 ms and 850 ms (diff = 100 ms > 50 ms)
    # Mean RR = 800 ms, RMSSD = 100 ms, pNN50 = 100%
    intervals = [750, 850, 750, 850, 750, 850]
    peak_indices = np.cumsum([0] + intervals)

    metrics = calculate_hrv_metrics(peak_indices, fs=fs)

    assert np.isclose(metrics.mean_rr, 800.0, atol=1.0)
    assert np.isclose(metrics.rmssd, 100.0, atol=1.0)
    assert np.isclose(metrics.pnn50, 100.0, atol=1e-2)
    assert metrics.sdnn > 0.0


def test_hrv_edge_cases():
    """Verify graceful handling when insufficient peaks are detected."""
    # Zero peaks
    m0 = calculate_hrv_metrics([], fs=30.0)
    assert len(m0.ibis) == 0
    assert m0.mean_rr == 0.0
    assert m0.sdnn == 0.0
    assert m0.rmssd == 0.0
    assert m0.pnn50 == 0.0
    assert m0.mean_hr == 0.0

    # Single peak (no interval)
    m1 = calculate_hrv_metrics([15], fs=30.0)
    assert len(m1.ibis) == 0
    assert m1.mean_rr == 0.0

    # Two peaks (single interval -> sdnn and rmssd are 0.0)
    m2 = calculate_hrv_metrics([10, 34], fs=30.0)
    assert len(m2.ibis) == 1
    assert np.isclose(m2.mean_rr, 800.0, atol=1.0)
    assert m2.sdnn == 0.0
    assert m2.rmssd == 0.0
    assert m2.pnn50 == 0.0

    # detect_peaks with short or constant signal
    assert len(detect_peaks([], fs=30.0)) == 0
    assert len(detect_peaks([1.0, 1.0], fs=30.0)) == 0
    assert len(detect_peaks(np.ones(50), fs=30.0)) == 0
