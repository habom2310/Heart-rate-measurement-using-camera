"""Core rPPG pulse extraction algorithms.

Implements POS (Plane-Orthogonal-to-Skin), CHROM (Chrominance),
and Green channel baseline methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

from rppg.core.signal import butter_bandpass_filter


class RPPGAlgorithm(ABC):
    """Abstract base class for rPPG pulse extraction algorithms."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize the algorithm with an optional descriptive name."""
        self.name = name or self.__class__.__name__

    @abstractmethod
    def process(
        self,
        rgb_buffer: np.ndarray | Sequence[Sequence[float]],
        fps: float = 30.0,
        filter_pulse: bool = True,
    ) -> np.ndarray:
        """Process RGB color buffer into a 1D pulse signal.

        Args:
            rgb_buffer: Array-like of shape (N, 3) representing RGB channels over time.
            fps: Video frame rate / sampling frequency in Hz (default: 30.0).
            filter_pulse: Whether to apply bandpass filtering (0.75 - 3.5 Hz).

        Returns:
            np.ndarray: 1D pulse signal of shape (N,).
        """
        pass

    def __call__(
        self,
        rgb_buffer: np.ndarray | Sequence[Sequence[float]],
        fps: float = 30.0,
        filter_pulse: bool = True,
    ) -> np.ndarray:
        """Callable shorthand for process()."""
        return self.process(rgb_buffer, fps=fps, filter_pulse=filter_pulse)


class POSAlgorithm(RPPGAlgorithm):
    """Plane-Orthogonal-to-Skin (POS) rPPG algorithm (Wang et al. 2017).

    Projects temporally normalized RGB signals onto two orthogonal chrominance
    planes and linearly combines them using standard deviations to isolate
    the blood volume pulse (BVP).
    """

    def process(
        self,
        rgb_buffer: np.ndarray | Sequence[Sequence[float]],
        fps: float = 30.0,
        filter_pulse: bool = True,
    ) -> np.ndarray:
        """Extract pulse signal using the POS algorithm.

        Closed-form projection:
            cn = C / mean(C)
            S1 = cn[:, 1] - cn[:, 2] (Green - Blue)
            S2 = cn[:, 1] + cn[:, 2] - 2 * cn[:, 0] (Green + Blue - 2 * Red)
            alpha = std(S1) / std(S2)
            pulse = S1 + alpha * S2
        """
        arr = np.asarray(rgb_buffer, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) == 0:
            return np.zeros(len(arr), dtype=float)

        mean_c = np.mean(arr, axis=0, keepdims=True)
        mean_c = np.where(np.abs(mean_c) < 1e-8, 1.0, mean_c)
        cn = arr / mean_c

        s1 = cn[:, 1] - cn[:, 2]
        s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]

        std_s2 = float(np.std(s2))
        if std_s2 > 1e-8:
            alpha = float(np.std(s1)) / std_s2
        else:
            alpha = 0.0

        pulse = s1 + alpha * s2

        if filter_pulse:
            return butter_bandpass_filter(pulse, lowcut=0.75, highcut=3.5, fs=fps)
        return pulse


class CHROMAlgorithm(RPPGAlgorithm):
    """Chrominance-based (CHROM) rPPG algorithm (de Haan & Jeanne 2013).

    Projects temporally normalized RGB signals into skin chrominance space (Xs, Ys)
    and combines them to suppress specular and motion artifacts.
    """

    def process(
        self,
        rgb_buffer: np.ndarray | Sequence[Sequence[float]],
        fps: float = 30.0,
        filter_pulse: bool = True,
    ) -> np.ndarray:
        """Extract pulse signal using the CHROM algorithm.

        Closed-form projection:
            cn = C / mean(C)
            X = 3 * R - 2 * G
            Y = 1.5 * R + G - 1.5 * B
            alpha = std(X) / std(Y)
            pulse = X - alpha * Y
        """
        arr = np.asarray(rgb_buffer, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3 or len(arr) == 0:
            return np.zeros(len(arr), dtype=float)

        mean_c = np.mean(arr, axis=0, keepdims=True)
        mean_c = np.where(np.abs(mean_c) < 1e-8, 1.0, mean_c)
        cn = arr / mean_c

        r = cn[:, 0]
        g = cn[:, 1]
        b = cn[:, 2]

        x = 3.0 * r - 2.0 * g
        y = 1.5 * r + g - 1.5 * b

        std_y = float(np.std(y))
        if std_y > 1e-8:
            alpha = float(np.std(x)) / std_y
        else:
            alpha = 0.0

        pulse = x - alpha * y

        if filter_pulse:
            return butter_bandpass_filter(pulse, lowcut=0.75, highcut=3.5, fs=fps)
        return pulse


class GreenMeanAlgorithm(RPPGAlgorithm):
    """Legacy baseline returning normalized green channel.

    Green light experiences the highest hemoglobin absorption modulation in
    ambient light photoplethysmography.
    """

    def process(
        self,
        rgb_buffer: np.ndarray | Sequence[Sequence[float]],
        fps: float = 30.0,
        filter_pulse: bool = True,
    ) -> np.ndarray:
        """Extract pulse signal using green channel normalization."""
        arr = np.asarray(rgb_buffer, dtype=float)
        if arr.size == 0 or len(arr) == 0:
            return np.zeros(len(arr), dtype=float)

        if arr.ndim == 2 and arr.shape[1] >= 2:
            g = arr[:, 1]
        elif arr.ndim == 1:
            g = arr
        else:
            return np.zeros(len(arr), dtype=float)

        mean_g = float(np.mean(g))
        if abs(mean_g) > 1e-8:
            gn = (g / mean_g) - 1.0
        else:
            gn = g - mean_g

        if filter_pulse:
            return butter_bandpass_filter(gn, lowcut=0.75, highcut=3.5, fs=fps)
        return gn
