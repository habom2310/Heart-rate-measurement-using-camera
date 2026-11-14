"""Unit tests for rPPG pulse extraction algorithms (POS, CHROM, Green baseline)."""

import numpy as np
import pytest

from rppg.core.algorithms import (
    CHROMAlgorithm,
    GreenMeanAlgorithm,
    POSAlgorithm,
)
from rppg.core.signal import estimate_heart_rate


@pytest.fixture
def synthetic_rppg_data():
    """Generate synthetic RGB signals with simulated PPG blood volume modulation.

    Simulates physiological modulation at 1.25 Hz (75 BPM) with realistic
    differing absorption across color channels (green absorption is highest).
    """
    fs = 30.0
    duration = 10.0  # 300 samples
    t = np.arange(0, duration, 1.0 / fs)
    cardiac_freq = 1.25  # 75 BPM

    pulse = np.sin(2 * np.pi * cardiac_freq * t)

    # Base skin reflection values
    r_base = 150.0
    g_base = 100.0
    b_base = 80.0

    # Hemoglobin absorption modulation: strongest in Green, weaker in Red/Blue
    r = r_base * (1.0 + 0.010 * pulse)
    g = g_base * (1.0 + 0.035 * pulse)
    b = b_base * (1.0 + 0.015 * pulse)

    # Add subtle white noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.05, size=(len(t), 3))
    rgb = np.column_stack([r, g, b]) + noise

    return rgb, fs, cardiac_freq * 60.0


def test_pos_algorithm(synthetic_rppg_data):
    """Verify POS algorithm extracts 1D pulse and accurately estimates heart rate."""
    rgb, fs, expected_bpm = synthetic_rppg_data
    algo = POSAlgorithm()

    # Process filtered pulse
    pulse = algo.process(rgb, fps=fs)
    assert isinstance(pulse, np.ndarray)
    assert pulse.ndim == 1
    assert len(pulse) == len(rgb)

    # Heart rate estimated from POS output should match simulated BPM
    hr = estimate_heart_rate(pulse, fs=fs)
    assert abs(hr.bpm - expected_bpm) < 2.0
    assert hr.snr > 0.0

    # Test callable shorthand
    pulse_call = algo(rgb, fps=fs)
    assert np.allclose(pulse, pulse_call)

    # Unfiltered mode
    unfiltered = algo.process(rgb, fps=fs, filter_pulse=False)
    assert len(unfiltered) == len(rgb)


def test_chrom_algorithm(synthetic_rppg_data):
    """Verify CHROM algorithm extracts 1D pulse and accurately estimates heart rate."""
    rgb, fs, expected_bpm = synthetic_rppg_data
    algo = CHROMAlgorithm()

    pulse = algo.process(rgb, fps=fs)
    assert isinstance(pulse, np.ndarray)
    assert pulse.ndim == 1
    assert len(pulse) == len(rgb)

    hr = estimate_heart_rate(pulse, fs=fs)
    assert abs(hr.bpm - expected_bpm) < 2.0
    assert hr.snr > 0.0

    # Unfiltered mode
    unfiltered = algo.process(rgb, fps=fs, filter_pulse=False)
    assert len(unfiltered) == len(rgb)


def test_green_mean_algorithm(synthetic_rppg_data):
    """Verify GreenMeanAlgorithm extracts pulse from 2D RGB or 1D green channel."""
    rgb, fs, expected_bpm = synthetic_rppg_data
    algo = GreenMeanAlgorithm()

    # 2D input
    pulse_2d = algo.process(rgb, fps=fs)
    assert isinstance(pulse_2d, np.ndarray)
    assert pulse_2d.ndim == 1
    assert len(pulse_2d) == len(rgb)

    hr = estimate_heart_rate(pulse_2d, fs=fs)
    assert abs(hr.bpm - expected_bpm) < 2.0

    # 1D green channel input
    pulse_1d = algo.process(rgb[:, 1], fps=fs)
    assert np.allclose(pulse_2d, pulse_1d)


def test_algorithm_edge_cases():
    """Verify edge case handling for algorithms: empty, short, or constant input."""
    for algo_cls in (POSAlgorithm, CHROMAlgorithm, GreenMeanAlgorithm):
        algo = algo_cls()

        # Empty input
        empty = algo.process(np.empty((0, 3)))
        assert len(empty) == 0

        # Short input
        short_res = algo.process([[100, 100, 100], [101, 101, 101]])
        assert len(short_res) == 2

        # Constant input (zero variance)
        const_rgb = np.ones((50, 3)) * 100.0
        const_res = algo.process(const_rgb)
        assert len(const_res) == 50
        assert not np.any(np.isnan(const_res))
