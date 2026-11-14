"""Unit tests for signal processing utilities in rppg.core.signal."""

import threading
import numpy as np
import pytest

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


def test_butter_bandpass_design():
    """Verify Butterworth bandpass coefficients and parameter validation."""
    b, a = butter_bandpass(lowcut=0.75, highcut=3.5, fs=30.0, order=3)
    assert len(b) > 0
    assert len(a) > 0
    assert len(b) == len(a)

    # Invalid frequency bounds should raise ValueError
    with pytest.raises(ValueError):
        butter_bandpass(lowcut=3.5, highcut=0.75, fs=30.0)

    with pytest.raises(ValueError):
        butter_bandpass(lowcut=0.75, highcut=16.0, fs=30.0)  # > Nyquist (15.0 Hz)


def test_butter_bandpass_filter():
    """Verify bandpass filtering attenuates out-of-band noise while passing pulse."""
    fs = 30.0
    t = np.arange(0, 10.0, 1.0 / fs)

    # 0.2 Hz low drift + 1.25 Hz cardiac (75 BPM) + 10.0 Hz high frequency noise
    low_drift = 2.0 * np.sin(2 * np.pi * 0.2 * t)
    pulse = 1.0 * np.sin(2 * np.pi * 1.25 * t)
    high_noise = 0.8 * np.sin(2 * np.pi * 10.0 * t)
    composite = low_drift + pulse + high_noise

    filtered = butter_bandpass_filter(composite, lowcut=0.75, highcut=3.5, fs=fs, order=3)

    assert filtered.shape == composite.shape

    # Attenuation check: the filtered signal should correlate strongly with pure pulse
    # Skip filter boundary transients (first and last 1 second)
    valid_slice = slice(int(fs), -int(fs))
    corr = np.corrcoef(pulse[valid_slice], filtered[valid_slice])[0, 1]
    assert corr > 0.95

    # Short signal fallback (< padlen)
    short_sig = np.array([1.0, 2.0, 1.5, 1.2])
    short_filtered = butter_bandpass_filter(short_sig, fs=fs)
    assert len(short_filtered) == len(short_sig)

    # Empty array
    assert len(butter_bandpass_filter([])) == 0


def test_butter_bandpass_filter_2d():
    """Verify 2D multi-channel filtering preserves shape."""
    fs = 30.0
    t = np.arange(0, 5.0, 1.0 / fs)
    sig_2d = np.column_stack([
        np.sin(2 * np.pi * 1.25 * t),
        np.cos(2 * np.pi * 1.25 * t),
        np.sin(2 * np.pi * 1.25 * t + np.pi / 4),
    ])
    filtered_2d = butter_bandpass_filter(sig_2d, fs=fs)
    assert filtered_2d.shape == sig_2d.shape


def test_detrend_signal():
    """Verify linear baseline trend is removed."""
    t = np.linspace(0, 10, 300)
    trend = 3.5 * t + 10.0
    pulse = np.sin(2 * np.pi * 1.25 * t)
    signal = trend + pulse

    detrended = detrend_signal(signal)
    assert np.allclose(np.mean(detrended), 0.0, atol=1e-2)
    # The remaining variance should be dominated by the pulse (~0.5 variance for unit sine)
    assert np.isclose(np.var(detrended), np.var(pulse), atol=0.05)


def test_interpolate_signal():
    """Verify resampling irregular timestamps to uniform grid."""
    np.random.seed(42)
    fs_target = 30.0
    duration = 5.0

    # Create irregular timestamps with jitter
    steps = np.random.uniform(0.025, 0.045, size=150)
    t_irregular = np.cumsum(steps)
    t_irregular -= t_irregular[0]
    t_irregular = t_irregular[t_irregular <= duration]

    sig = np.sin(2 * np.pi * 1.25 * t_irregular)

    # Linear interpolation
    interp_sig, even_times = interpolate_signal(sig, t_irregular, target_fs=fs_target, method="linear")
    dt = np.diff(even_times)
    expected_dt = (even_times[-1] - even_times[0]) / (len(even_times) - 1)
    assert np.allclose(dt, expected_dt, atol=1e-5)
    assert np.isclose(expected_dt, 1.0 / fs_target, atol=1e-3)

    # Cubic interpolation
    cubic_sig, _ = interpolate_signal(sig, t_irregular, target_fs=fs_target, method="cubic")
    assert len(cubic_sig) == len(interp_sig)

    # return_times=False option
    sig_only = interpolate_signal(sig, t_irregular, target_fs=fs_target, return_times=False)
    assert np.allclose(sig_only, interp_sig)


def test_compute_welch_psd():
    """Verify Welch PSD computation identifies signal peak."""
    fs = 30.0
    t = np.arange(0, 10.0, 1.0 / fs)
    sig = np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz = 90 BPM

    freqs, psd = compute_welch_psd(sig, fs=fs)
    assert len(freqs) == len(psd)
    peak_freq = freqs[np.argmax(psd)]
    assert np.isclose(peak_freq, 1.5, atol=0.05)


def test_estimate_heart_rate():
    """Test estimate_heart_rate with synthetic 1.25 Hz (75 BPM) signal plus noise.

    Requirement: recovered BPM must be within 1.0 BPM of 75.0 BPM and SNR > 0 dB.
    """
    np.random.seed(123)
    fs = 30.0
    duration = 10.0
    t = np.arange(0, duration, 1.0 / fs)

    # 1.25 Hz = 75 BPM sine wave plus noise
    cardiac_freq = 1.25
    expected_bpm = cardiac_freq * 60.0  # 75.0 BPM
    clean_pulse = np.sin(2 * np.pi * cardiac_freq * t)
    noise = 0.15 * np.random.normal(size=len(t))
    signal_data = clean_pulse + noise

    res = estimate_heart_rate(signal_data, fs=fs, min_bpm=45.0, max_bpm=210.0)

    assert isinstance(res, HeartRateResult)
    assert abs(res.bpm - expected_bpm) < 1.0
    assert res.snr > 0.0

    # Verify tuple unpacking support
    bpm, snr = estimate_heart_rate(signal_data, fs=fs)
    assert bpm == res.bpm
    assert snr == res.snr

    # Edge cases
    assert estimate_heart_rate([], fs=fs).bpm == 0.0
    assert estimate_heart_rate([1.0, 2.0], fs=fs).bpm == 0.0


def test_signal_buffer():
    """Verify SignalBuffer ring buffer operations and thread safety."""
    buf = SignalBuffer(capacity=5)
    assert len(buf) == 0
    assert not buf.is_full()

    # Append frames
    for i in range(5):
        buf.append([float(i), float(i * 2), float(i * 3)], timestamp=float(i))
    assert len(buf) == 5
    assert buf.is_full()

    # Verify FIFO overflow drops oldest
    buf.append([10.0, 20.0, 30.0], timestamp=5.0)
    assert len(buf) == 5
    rgb, ts = buf.get_data()
    assert ts[0] == 1.0  # index 0 was dropped
    assert ts[-1] == 5.0
    assert np.allclose(rgb[-1], [10.0, 20.0, 30.0])

    # Direct accessors
    assert len(buf.get_rgb()) == 5
    assert len(buf.get_timestamps()) == 5

    # Interpolation from buffer
    interp_rgb, interp_ts = buf.get_interpolated(target_fs=30.0)
    assert len(interp_rgb) == len(interp_ts)

    # Clear
    buf.clear()
    assert len(buf) == 0
    assert not buf.is_full()


def test_signal_buffer_thread_safety():
    """Verify concurrent appends to SignalBuffer do not crash or corrupt."""
    buf = SignalBuffer(capacity=200)

    def producer(start_idx: int):
        for i in range(100):
            buf.append([float(start_idx + i)] * 3, timestamp=float(i * 0.033))

    threads = [
        threading.Thread(target=producer, args=(0,)),
        threading.Thread(target=producer, args=(1000,)),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert len(buf) == 200
    rgb, ts = buf.get_data()
    assert rgb.shape == (200, 3)
    assert len(ts) == 200
