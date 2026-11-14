"""Eulerian Video Magnification for subtle color variations and motion.

Adapted from Wu et al. (2012) Eulerian Video Magnification for Revealing
Subtle Changes in the World.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Union

import cv2
import numpy as np
from scipy import fftpack, signal


def resize_width(image: np.ndarray, width: int) -> np.ndarray:
    """Resize image preserving aspect ratio given a target width."""
    h, w = image.shape[:2]
    new_h = max(1, int(h * (width / float(w))))
    return cv2.resize(image, (width, new_h), interpolation=cv2.INTER_AREA)


class VideoMagnifier:
    """Performs Eulerian Color or Motion magnification on video frames."""

    def __init__(
        self,
        buffer_size: int = 40,
        levels: int = 3,
        amplification: float = 30.0,
        low_freq: float = 0.4,
        high_freq: float = 2.0,
        mode: str = "color",
    ) -> None:
        self.buffer_size = buffer_size
        self.levels = levels
        self.amplification = amplification
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.mode = mode

        self.fps: float = 0.0
        self.times: List[float] = []
        self.t0: float = time.time()
        self.data_buffer: List[np.ndarray] = []
        self.frame_out: np.ndarray = np.zeros((10, 10, 3), dtype=np.uint8)

    # ------------------ Color Magnification ------------------ #

    @staticmethod
    def build_gaussian_pyramid(src: np.ndarray, level: int = 3) -> List[np.ndarray]:
        """Build Gaussian pyramid up to level."""
        current = src.copy()
        pyramid = [current]
        for _ in range(level):
            current = cv2.pyrDown(current)
            pyramid.append(current)
        return pyramid

    def gaussian_video(self, video_tensor: np.ndarray, levels: int = 3) -> np.ndarray:
        """Extract top of Gaussian pyramid for every frame in tensor."""
        n_frames = video_tensor.shape[0]
        vid_data = None
        for i in range(n_frames):
            pyr = self.build_gaussian_pyramid(video_tensor[i], level=levels)
            gaussian_frame = pyr[-1]
            if vid_data is None:
                vid_data = np.zeros(
                    (n_frames, gaussian_frame.shape[0], gaussian_frame.shape[1], 3),
                    dtype=np.float64,
                )
            vid_data[i] = gaussian_frame
        return vid_data if vid_data is not None else np.empty(0)

    @staticmethod
    def temporal_ideal_filter(
        tensor: np.ndarray, low: float, high: float, fps: float, axis: int = 0
    ) -> np.ndarray:
        """Apply ideal temporal bandpass filter via FFT."""
        fft_data = fftpack.fft(tensor, axis=axis)
        frequencies = fftpack.fftfreq(tensor.shape[axis], d=1.0 / fps)
        bound_low = int(np.abs(frequencies - low).argmin())
        bound_high = int(np.abs(frequencies - high).argmin())

        fft_data[:bound_low] = 0
        if bound_high > 0:
            fft_data[bound_high:-bound_high] = 0
        if bound_low > 0:
            fft_data[-bound_low:] = 0

        iff = fftpack.ifft(fft_data, axis=axis)
        return np.abs(iff)

    @staticmethod
    def reconstruct_video(amp_video: np.ndarray, origin_video: np.ndarray, levels: int = 3) -> np.ndarray:
        """Reconstruct full-resolution frames by upsampling and adding to original."""
        final_video = np.zeros_like(origin_video)
        for i in range(amp_video.shape[0]):
            img = amp_video[i]
            for _ in range(levels):
                img = cv2.pyrUp(img)
            # Ensure shapes match after pyrUp rounding
            h_orig, w_orig = origin_video[i].shape[:2]
            if img.shape[:2] != (h_orig, w_orig):
                img = cv2.resize(img, (w_orig, h_orig))
            final_video[i] = img + origin_video[i]
        return final_video

    def magnify_color(self, data_buffer: np.ndarray, fps: float) -> np.ndarray:
        """Eulerian color magnification pipeline."""
        gau_video = self.gaussian_video(data_buffer, levels=self.levels)
        filtered = self.temporal_ideal_filter(
            gau_video, self.low_freq, self.high_freq, fps
        )
        amplified = filtered * self.amplification
        return self.reconstruct_video(amplified, data_buffer, levels=self.levels)

    # ------------------ Motion Magnification ------------------ #

    def build_laplacian_pyramid(self, src: np.ndarray, levels: int = 3) -> List[np.ndarray]:
        """Build Laplacian pyramid."""
        gaussian_pyramid = self.build_gaussian_pyramid(src, levels)
        pyramid = []
        for i in range(levels, 0, -1):
            ge = cv2.pyrUp(gaussian_pyramid[i])
            h, w = gaussian_pyramid[i - 1].shape[:2]
            if ge.shape[:2] != (h, w):
                ge = cv2.resize(ge, (w, h))
            lap = cv2.subtract(gaussian_pyramid[i - 1], ge)
            pyramid.append(lap)
        return pyramid

    def laplacian_video(self, video_tensor: np.ndarray, levels: int = 3) -> List[np.ndarray]:
        """Compute Laplacian pyramid tensor list for a video sequence."""
        n_frames = video_tensor.shape[0]
        pyr_list: List[List[np.ndarray]] = []
        for i in range(n_frames):
            pyr = self.build_laplacian_pyramid(video_tensor[i], levels=levels)
            pyr_list.append(pyr)

        tensor_list = []
        for n in range(levels):
            level_tensor = np.zeros(
                (n_frames, pyr_list[0][n].shape[0], pyr_list[0][n].shape[1], 3),
                dtype=np.float64,
            )
            for i in range(n_frames):
                level_tensor[i] = pyr_list[i][n]
            tensor_list.append(level_tensor)
        return tensor_list

    @staticmethod
    def butter_bandpass_filter(
        data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5
    ) -> np.ndarray:
        """Apply Butterworth bandpass filter along axis 0."""
        omega = 0.5 * fs
        low = max(0.001, min(0.999, lowcut / omega))
        high = max(0.001, min(0.999, highcut / omega))
        if low >= high:
            return data
        b, a = signal.butter(order, [low, high], btype="band")
        return signal.lfilter(b, a, data, axis=0)

    @staticmethod
    def reconstruct_from_tensorlist(filter_tensor_list: List[np.ndarray], levels: int = 3) -> np.ndarray:
        """Reconstruct video from Laplacian filtered pyramid."""
        n_frames = filter_tensor_list[0].shape[0]
        final = np.zeros_like(filter_tensor_list[-1])
        for i in range(n_frames):
            up = filter_tensor_list[0][i]
            for n in range(levels - 1):
                up = cv2.pyrUp(up)
                h, w = filter_tensor_list[n + 1][i].shape[:2]
                if up.shape[:2] != (h, w):
                    up = cv2.resize(up, (w, h))
                up = up + filter_tensor_list[n + 1][i]
            final[i] = up
        return final

    def magnify_motion(self, video_tensor: np.ndarray, fps: float) -> np.ndarray:
        """Eulerian motion magnification pipeline."""
        lap_video_list = self.laplacian_video(video_tensor, levels=self.levels)
        filter_tensor_list = []
        for i in range(self.levels):
            filt = self.butter_bandpass_filter(
                lap_video_list[i], self.low_freq, self.high_freq, fps
            )
            filt = filt * self.amplification
            filter_tensor_list.append(filt)
        recon = self.reconstruct_from_tensorlist(filter_tensor_list, levels=self.levels)
        return video_tensor + recon

    # ------------------ Processing & Loop ------------------ #

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Add frame to buffer and run amplification when buffer is full."""
        resized = resize_width(frame, 256)
        # Ensure dimensions match standard shape (e.g. 192x256)
        if resized.shape[0] != 192:
            resized = cv2.resize(resized, (256, 192), interpolation=cv2.INTER_AREA)

        self.times.append(time.time() - self.t0)
        self.data_buffer.append(resized.astype(np.float64))

        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]

        if len(self.data_buffer) == self.buffer_size:
            dt = self.times[-1] - self.times[0]
            if dt > 0:
                self.fps = float(self.buffer_size) / dt
            if self.fps > 5.0:
                tensor = np.array(self.data_buffer, dtype=np.float64)
                if self.mode == "motion":
                    final_vid = self.magnify_motion(tensor, fps=self.fps)
                else:
                    final_vid = self.magnify_color(tensor, fps=self.fps)
                self.frame_out = np.clip(final_vid[-1], 0, 255).astype(np.uint8)

        return self.frame_out


def run(
    source: Union[int, str] = 0,
    mode: str = "color",
    amplification: float = 30.0,
    low: float = 0.4,
    high: float = 2.0,
    buffer_size: int = 40,
) -> None:
    """Run Eulerian magnification loop from webcam or video file."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        sys.exit(1)

    is_webcam = isinstance(source, int)
    magnifier = VideoMagnifier(
        buffer_size=buffer_size,
        levels=3,
        amplification=amplification,
        low_freq=low,
        high_freq=high,
        mode=mode,
    )

    print(f"[INFO] Started Eulerian Video Magnification ({mode} mode)")
    print("[INFO] Press 'q' or 'ESC' to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if is_webcam:
                frame = cv2.flip(frame, 1)

            magnified = magnifier.process_frame(frame)

            fps_text = f"FPS: {magnifier.fps:.1f}"
            cv2.putText(
                frame, fps_text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2
            )

            cv2.imshow("Input Video", frame)
            if magnified.shape[0] > 10:
                display_mag = resize_width(magnified, 640)
                cv2.imshow(f"Magnified ({mode})", display_mag)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Finished.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Eulerian Video Magnification for color pulse or motion."
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        default="0",
        help="Webcam index (e.g. 0) or video file path (default: 0)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["color", "motion"],
        default="color",
        help="Magnification mode: 'color' or 'motion' (default: color)",
    )
    parser.add_argument(
        "--amplification",
        "-a",
        type=float,
        default=30.0,
        help="Amplification factor (default: 30.0)",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=0.4,
        help="Low frequency bound in Hz (default: 0.4 Hz = 24 BPM)",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=2.0,
        help="High frequency bound in Hz (default: 2.0 Hz = 120 BPM)",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=40,
        help="Sliding buffer size in frames (default: 40)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: Union[int, str]
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run(
        source=source,
        mode=args.mode,
        amplification=args.amplification,
        low=args.low,
        high=args.high,
        buffer_size=args.buffer_size,
    )


if __name__ == "__main__":
    main()
