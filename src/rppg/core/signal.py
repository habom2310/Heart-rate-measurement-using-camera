"""Signal processing utilities and filters for rPPG pulse extraction."""

from __future__ import annotations

import collections
import threading
import time
from typing import NamedTuple, Sequence

import numpy as np
import scipy.interpolate
import scipy.signal


class HeartRateResult(NamedTuple):
    """Estimated heart rate and signal quality metrics."""

    bpm: float
    snr: float


def butter_bandpass(
    lowcut: float = 0.75,
    highcut: float = 3.5,
    fs: float = 30.0,
    order: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Design a Butterworth bandpass filter for cardiac frequencies.

    Args:
        lowcut: Lower cutoff frequency in Hz (default: 0.75 Hz = 45 BPM).
        highcut: Upper cutoff frequency in Hz (default: 3.5 Hz = 210 BPM).
        fs: Sampling frequency in Hz (default: 30.0 Hz).
        order: Filter order (default: 3).

    Returns:
        tuple[np.ndarray, np.ndarray]: Filter numerator (b) and denominator (a) polynomials.
    """
    nyq = 0.5 * fs
    if nyq <= 0:
        raise ValueError(f"Sampling frequency fs must be positive, got {fs}")

    if lowcut <= 0:
        raise ValueError(f"lowcut must be positive, got {lowcut}")
    if highcut >= nyq:
        raise ValueError(
            f"highcut ({highcut} Hz) must be strictly below Nyquist frequency ({nyq} Hz)."
        )
    if lowcut >= highcut:
        raise ValueError(
            f"lowcut ({lowcut} Hz) must be strictly less than highcut ({highcut} Hz)."
        )

    low = lowcut / nyq
    high = highcut / nyq

    b, a = scipy.signal.butter(order, [low, high], btype="bandpass")
    return b, a


def butter_bandpass_filter(
    data: np.ndarray | Sequence[float],
    lowcut: float = 0.75,
    highcut: float = 3.5,
    fs: float = 30.0,
    order: int = 3,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter to 1D or 2D signals.

    Uses `scipy.signal.filtfilt` when length exceeds filter padding length
    for zero-phase distortion, falling back to `scipy.signal.lfilter` for shorter
    signals.

    Args:
        data: Input signal array of shape (N,) or (N, C).
        lowcut: Lower cutoff frequency in Hz (default: 0.75 Hz).
        highcut: Upper cutoff frequency in Hz (default: 3.5 Hz).
        fs: Sampling frequency in Hz (default: 30.0 Hz).
        order: Filter order (default: 3).

    Returns:
        np.ndarray: Filtered signal matching input shape.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0 or len(arr) == 0:
        return arr.copy()

    b, a = butter_bandpass(lowcut, highcut, fs, order)
    padlen = 3 * max(len(a), len(b))

    if len(arr) > padlen:
        try:
            return scipy.signal.filtfilt(b, a, arr, axis=0)
        except ValueError:
            return scipy.signal.lfilter(b, a, arr, axis=0)

    return scipy.signal.lfilter(b, a, arr, axis=0)


def interpolate_signal(
    signal_data: np.ndarray | Sequence[float],
    times: np.ndarray | Sequence[float],
    target_fs: float = 30.0,
    method: str = "linear",
    return_times: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Evenly resample irregular timestamped samples to a fixed sampling rate.

    Args:
        signal_data: 1D or 2D array of signal samples.
        times: Monotonically increasing timestamps in seconds.
        target_fs: Target uniform sampling rate in Hz (default: 30.0).
        method: Interpolation method, 'linear' or 'cubic' (default: 'linear').
        return_times: Whether to return the interpolated timestamps array.

    Returns:
        tuple[np.ndarray, np.ndarray] | np.ndarray: Resampled signal and uniform timestamps,
            or just resampled signal if `return_times=False`.
    """
    arr = np.asarray(signal_data, dtype=float)
    t = np.asarray(times, dtype=float)

    if len(t) < 2 or len(arr) < 2:
        return (arr.copy(), t.copy()) if return_times else arr.copy()

    duration = float(t[-1] - t[0])
    if duration <= 0:
        return (arr.copy(), t.copy()) if return_times else arr.copy()

    num_samples = max(2, int(np.round(duration * target_fs)) + 1)
    even_times = np.linspace(t[0], t[-1], num_samples)

    if method == "cubic" and len(t) >= 4:
        spline = scipy.interpolate.CubicSpline(t, arr, axis=0)
        interp = spline(even_times)
    else:
        if arr.ndim == 1:
            interp = np.interp(even_times, t, arr)
        else:
            interp_func = scipy.interpolate.interp1d(
                t, arr, axis=0, kind="linear", fill_value="extrapolate"
            )
            interp = interp_func(even_times)

    if return_times:
        return interp, even_times
    return interp


def detrend_signal(
    data: np.ndarray | Sequence[float],
    type: str = "linear",
) -> np.ndarray:
    """Remove baseline drift or linear trends from signal data.

    Args:
        data: 1D or 2D array of samples.
        type: Type of detrending, 'linear' or 'constant' (default: 'linear').

    Returns:
        np.ndarray: Detrended signal array.
    """
    arr = np.asarray(data, dtype=float)
    if arr.size == 0 or len(arr) == 0:
        return arr.copy()
    return scipy.signal.detrend(arr, axis=0, type=type)


def compute_welch_psd(
    signal_data: np.ndarray | Sequence[float],
    fs: float = 30.0,
    nperseg: int | None = None,
    nfft: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Power Spectral Density (PSD) using Welch's method.

    Args:
        signal_data: 1D signal samples.
        fs: Sampling frequency in Hz (default: 30.0).
        nperseg: Length of each segment (default: min(len(data), 256)).
        nfft: Length of FFT used (default: >= 2048 for high frequency resolution).

    Returns:
        tuple[np.ndarray, np.ndarray]: Frequencies array and PSD values array.
    """
    arr = np.asarray(signal_data, dtype=float)
    if len(arr) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    if nperseg is None:
        nperseg = min(len(arr), 256)
    else:
        nperseg = min(len(arr), max(4, nperseg))

    if nfft is None:
        nfft = max(2048, 2 ** int(np.ceil(np.log2(max(len(arr), 64) * 4))))

    freqs, psd = scipy.signal.welch(
        arr,
        fs=fs,
        nperseg=nperseg,
        nfft=nfft,
        axis=0,
    )
    return freqs, psd


def estimate_heart_rate(
    signal_data: np.ndarray | Sequence[float],
    fs: float = 30.0,
    min_bpm: float = 45.0,
    max_bpm: float = 210.0,
) -> HeartRateResult:
    """Estimate heart rate (BPM) and Signal-to-Noise Ratio (SNR in dB).

    Calculates dominant peak frequency using Welch PSD with parabolic sub-bin
    interpolation for sub-bin precision, and evaluates signal quality index (SNR).

    Args:
        signal_data: 1D pulse signal samples.
        fs: Sampling rate in Hz (default: 30.0).
        min_bpm: Minimum physiological heart rate in BPM (default: 45.0).
        max_bpm: Maximum physiological heart rate in BPM (default: 210.0).

    Returns:
        HeartRateResult: NamedTuple containing:
            - bpm: Estimated heart rate in beats per minute.
            - snr: Signal-to-noise ratio in decibels (dB).
    """
    arr = np.asarray(signal_data, dtype=float)
    if len(arr) < 4:
        return HeartRateResult(bpm=0.0, snr=-30.0)

    clean_signal = detrend_signal(arr)
    nfft = max(2048, 2 ** int(np.ceil(np.log2(len(clean_signal) * 4))))
    freqs, psd = compute_welch_psd(
        clean_signal,
        fs=fs,
        nperseg=len(clean_signal),
        nfft=nfft,
    )

    min_f = min_bpm / 60.0
    max_f = max_bpm / 60.0
    band_mask = (freqs >= min_f) & (freqs <= max_f)

    if not np.any(band_mask):
        return HeartRateResult(bpm=0.0, snr=-30.0)

    band_indices = np.where(band_mask)[0]
    sub_peak = int(np.argmax(psd[band_mask]))
    peak_idx = band_indices[sub_peak]

    # Parabolic sub-bin interpolation
    if 0 < peak_idx < len(psd) - 1:
        alpha = float(psd[peak_idx - 1])
        beta = float(psd[peak_idx])
        gamma = float(psd[peak_idx + 1])
        denom = 2.0 * (2.0 * beta - alpha - gamma)
        if denom > 1e-12:
            p = (gamma - alpha) / denom
            p = max(-0.5, min(0.5, p))
        else:
            p = 0.0
        df = freqs[1] - freqs[0]
        peak_f = float(freqs[peak_idx] + p * df)
    else:
        peak_f = float(freqs[peak_idx])

    bpm = float(peak_f * 60.0)
    bpm = max(min_bpm, min(max_bpm, bpm))

    # Signal-to-Noise Ratio (SNR) in dB
    # Signal power: power within fundamental +/- 0.15 Hz and 2nd harmonic +/- 0.15 Hz
    delta_f = 0.15
    f_fund = peak_f
    f_harm = 2.0 * peak_f

    sig_mask = ((freqs >= f_fund - delta_f) & (freqs <= f_fund + delta_f))
    if f_harm <= max_f:
        sig_mask |= ((freqs >= f_harm - delta_f) & (freqs <= f_harm + delta_f))

    sig_in_band = sig_mask & band_mask
    noise_in_band = (~sig_mask) & band_mask

    p_sig = float(np.sum(psd[sig_in_band]))
    p_noise = float(np.sum(psd[noise_in_band]))

    if p_noise <= 1e-12:
        snr = 30.0
    elif p_sig <= 1e-12:
        snr = -30.0
    else:
        snr = float(10.0 * np.log10(p_sig / p_noise))

    return HeartRateResult(bpm=round(bpm, 2), snr=round(snr, 2))


class SignalBuffer:
    """Thread-safe circular ring buffer for storing raw RGB values and timestamps."""

    def __init__(self, capacity: int = 300) -> None:
        """Initialize buffer with given maximum capacity.

        Args:
            capacity: Maximum number of frames to retain (default: 300).
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._capacity = capacity
        self._lock = threading.Lock()
        self._timestamps: collections.deque[float] = collections.deque(maxlen=capacity)
        self._rgb_data: collections.deque[np.ndarray] = collections.deque(maxlen=capacity)

    @property
    def capacity(self) -> int:
        """Maximum number of frames buffer can store."""
        return self._capacity

    def append(
        self,
        rgb: Sequence[float] | np.ndarray,
        timestamp: float | None = None,
    ) -> None:
        """Append an RGB measurement with an optional timestamp.

        Args:
            rgb: Sequence of 3 floats representing (R, G, B) means.
            timestamp: Timestamp in seconds (default: current time.time()).
        """
        if timestamp is None:
            timestamp = time.time()

        arr = np.asarray(rgb, dtype=float).ravel()
        if len(arr) != 3:
            raise ValueError(f"Expected 3 color channels (R, G, B), got {len(arr)}")

        with self._lock:
            self._timestamps.append(float(timestamp))
            self._rgb_data.append(arr)

    def add(
        self,
        rgb: Sequence[float] | np.ndarray,
        timestamp: float | None = None,
    ) -> None:
        """Alias for append."""
        self.append(rgb, timestamp)

    def push(
        self,
        rgb: Sequence[float] | np.ndarray,
        timestamp: float | None = None,
    ) -> None:
        """Alias for append."""
        self.append(rgb, timestamp)

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve stored RGB values and timestamps as numpy arrays.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - rgb_array of shape (N, 3)
                - timestamps array of shape (N,)
        """
        with self._lock:
            if not self._timestamps:
                return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
            return (
                np.array(self._rgb_data, dtype=float),
                np.array(self._timestamps, dtype=float),
            )

    def get_rgb(self) -> np.ndarray:
        """Retrieve stored RGB values array of shape (N, 3)."""
        with self._lock:
            if not self._rgb_data:
                return np.empty((0, 3), dtype=float)
            return np.array(self._rgb_data, dtype=float)

    def get_timestamps(self) -> np.ndarray:
        """Retrieve stored timestamps array of shape (N,)."""
        with self._lock:
            if not self._timestamps:
                return np.empty(0, dtype=float)
            return np.array(self._timestamps, dtype=float)

    def get_interpolated(
        self,
        target_fs: float = 30.0,
        method: str = "linear",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resample stored buffer to a fixed sampling rate.

        Args:
            target_fs: Target uniform sampling rate in Hz (default: 30.0).
            method: Interpolation method ('linear' or 'cubic').

        Returns:
            tuple[np.ndarray, np.ndarray]: Resampled RGB array and uniform timestamps.
        """
        rgb, ts = self.get_data()
        if len(ts) < 2:
            return rgb, ts
        interp_rgb, interp_ts = interpolate_signal(  # type: ignore[misc]
            rgb, ts, target_fs=target_fs, method=method, return_times=True
        )
        return interp_rgb, interp_ts

    def clear(self) -> None:
        """Clear all stored samples."""
        with self._lock:
            self._timestamps.clear()
            self._rgb_data.clear()

    def is_full(self) -> bool:
        """Check if buffer is filled to capacity."""
        with self._lock:
            return len(self._timestamps) >= self._capacity

    def __len__(self) -> int:
        """Current number of stored samples."""
        with self._lock:
            return len(self._timestamps)
