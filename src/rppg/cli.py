"""Command-line interface (CLI) for real-time rPPG heart rate monitoring."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Optional

import cv2
import numpy as np

from rppg.core.algorithms import CHROMAlgorithm, GreenMeanAlgorithm, POSAlgorithm, RPPGAlgorithm
from rppg.core.tracker import BaseTracker, MediaPipeTracker, YuNetTracker
from rppg.pipeline import PipelineResult, RPPGPipeline


def _draw_pulse_waveform(
    frame: np.ndarray,
    pulse_wave: np.ndarray,
    width: int = 240,
    height: int = 70,
) -> np.ndarray:
    """Render a compact real-time pulse waveform inset graph."""
    if len(pulse_wave) < 2:
        return frame

    h, w = frame.shape[:2]
    x0 = w - width - 15
    y0 = h - height - 15
    if x0 < 0 or y0 < 0:
        return frame

    # Translucent card background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + width, y0 + height), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0.0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (60, 60, 70), 1)

    recent = pulse_wave[-width:]
    min_v, max_v = float(np.min(recent)), float(np.max(recent))
    v_range = max_v - min_v
    if v_range < 1e-6:
        v_range = 1.0

    n = len(recent)
    pts = []
    for i, val in enumerate(recent):
        px = int(x0 + (i / max(1, n - 1)) * (width - 10) + 5)
        py = int(y0 + height - 6 - ((val - min_v) / v_range) * (height - 12))
        pts.append([px, py])

    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts_arr], isClosed=False, color=(0, 255, 136), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(
        frame,
        "PULSE WAVE",
        (x0 + 6, y0 + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return frame


def draw_hud(frame: np.ndarray, result: PipelineResult) -> np.ndarray:
    """Render modern HUD telemetry card and waveform onto video frame."""
    h, w = frame.shape[:2]
    card_w, card_h = 240, 105
    x0, y0 = 15, 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + card_w, y0 + card_h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0.0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x0 + card_w, y0 + card_h), (60, 60, 70), 1)

    # Title
    cv2.putText(
        frame,
        "rPPG MONITOR",
        (x0 + 12, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Heart Rate Value
    if result.bpm is not None:
        bpm_str = f"{result.bpm:.0f}"
        bpm_color = (0, 255, 136) if (result.snr is not None and result.snr >= 0) else (0, 200, 255)
    else:
        bpm_str = "--"
        bpm_color = (150, 150, 150)

    cv2.putText(frame, bpm_str, (x0 + 12, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 1.2, bpm_color, 2, cv2.LINE_AA)
    cv2.putText(frame, "BPM", (x0 + 95, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    # Metrics line
    snr_str = f"SNR: {result.snr:+.1f}dB" if result.snr is not None else "SNR: --"
    fps_str = f"FPS: {result.fps:.1f}"
    status_str = "TRACKING" if result.face_detected else "SEARCHING"
    status_color = (0, 255, 136) if result.face_detected else (0, 70, 255)

    cv2.putText(
        frame,
        f"{fps_str}  |  {snr_str}",
        (x0 + 12, y0 + 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status_str,
        (x0 + 12, y0 + 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        status_color,
        1,
        cv2.LINE_AA,
    )

    if len(result.pulse_wave) > 0:
        _draw_pulse_waveform(frame, result.pulse_wave)

    return frame


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for rPPG runner."""
    parser = argparse.ArgumentParser(
        prog="rppg",
        description="Real-time remote photoplethysmography (rPPG) heart rate estimation.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source index (e.g. 0 for webcam) or video file path (default: 0).",
    )
    parser.add_argument(
        "--algo",
        choices=["pos", "chrom", "green"],
        default="pos",
        help="Pulse extraction algorithm to use (default: pos).",
    )
    parser.add_argument(
        "--tracker",
        choices=["auto", "mediapipe", "yunet"],
        default="auto",
        help="Face tracker backend: auto, mediapipe, or yunet (default: auto).",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=150,
        help="Size of sliding signal buffer in frames (default: 150).",
    )
    parser.add_argument(
        "--export",
        "--output",
        dest="export",
        type=str,
        default=None,
        help="File path to export measurement signals and metrics as CSV.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless without rendering OpenCV display window.",
    )
    return parser.parse_args(args)


def create_algorithm(name: str) -> RPPGAlgorithm:
    """Instantiate chosen rPPG algorithm."""
    name_lower = name.lower()
    if name_lower == "pos":
        return POSAlgorithm()
    elif name_lower == "chrom":
        return CHROMAlgorithm()
    elif name_lower == "green":
        return GreenMeanAlgorithm()
    else:
        raise ValueError(f"Unknown algorithm: {name}")


def create_tracker(name: str) -> BaseTracker:
    """Instantiate chosen tracker backend."""
    name_lower = name.lower()
    if name_lower == "mediapipe":
        return MediaPipeTracker()
    elif name_lower == "yunet":
        return YuNetTracker()
    elif name_lower == "auto":
        try:
            return MediaPipeTracker()
        except Exception:
            return YuNetTracker()
    else:
        raise ValueError(f"Unknown tracker: {name}")


def main(args: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parsed_args = parse_args(args)

    # Determine source (int device index vs file string)
    try:
        source: int | str = int(parsed_args.source)
    except ValueError:
        source = parsed_args.source

    algo = create_algorithm(parsed_args.algo)
    tracker = create_tracker(parsed_args.tracker)

    pipeline = RPPGPipeline(
        tracker=tracker,
        algorithm=algo,
        buffer_size=parsed_args.buffer_size,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {source}", file=sys.stderr)
        return 1

    csv_file = None
    csv_writer = None
    if parsed_args.export:
        try:
            csv_file = open(parsed_args.export, mode="w", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["timestamp", "r", "g", "b", "bpm", "snr", "fps"])
        except Exception as e:
            print(f"Warning: Failed to open export file {parsed_args.export}: {e}", file=sys.stderr)

    window_name = "rPPG Heart Rate Monitor - Press 'q' or 'ESC' to exit"
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            now = time.time()
            result = pipeline.process_frame(frame, timestamp=now)

            if csv_writer is not None:
                r, g, b = result.raw_rgb
                csv_writer.writerow(
                    [
                        f"{now:.4f}",
                        f"{r:.4f}",
                        f"{g:.4f}",
                        f"{b:.4f}",
                        f"{result.bpm:.2f}" if result.bpm is not None else "",
                        f"{result.snr:.2f}" if result.snr is not None else "",
                        f"{result.fps:.2f}",
                    ]
                )

            if not parsed_args.no_display:
                display_frame = draw_hud(result.frame, result)
                cv2.imshow(window_name, display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # 'q' or ESC
                    break
    finally:
        cap.release()
        if not parsed_args.no_display:
            cv2.destroyAllWindows()
        if csv_file is not None:
            csv_file.close()
        pipeline.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
