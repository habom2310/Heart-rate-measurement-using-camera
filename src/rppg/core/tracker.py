"""Face tracking and landmark detection modules for rPPG.

Provides BaseTracker interface with MediaPipe FaceMesh implementation and
OpenCV YuNet DNN fallback.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as err:
    raise ImportError("OpenCV is required for tracking. Run `pip install opencv-python`.") from err


@dataclass
class FaceLandmarksResult:
    """Detection result containing facial bounding box and landmark points.

    Attributes:
        bbox: Bounding box tuple (x, y, width, height) in pixel coordinates.
        landmarks: Landmark coordinates of shape (N, 2) in pixel space (x, y).
        normalized_landmarks: Landmark coordinates of shape (N, 2) normalized to [0, 1].
        confidence: Detection or tracking confidence score in [0.0, 1.0].
    """

    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray
    normalized_landmarks: np.ndarray
    confidence: float


class BaseTracker(ABC):
    """Abstract interface for face tracking and landmark extraction."""

    @abstractmethod
    def process(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarksResult]:
        """Detect and track facial landmarks in a BGR video frame.

        Args:
            frame_bgr: Video frame in BGR color space (H, W, 3).

        Returns:
            FaceLandmarksResult if a face is detected, None otherwise.
        """
        pass

    def close(self) -> None:
        """Release underlying detector resources if applicable."""
        pass

    def __enter__(self) -> BaseTracker:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class MediaPipeTracker(BaseTracker):
    """Face mesh tracker based on Google MediaPipe Face Mesh / Tasks FaceLandmarker.

    Extracts high-density facial mesh (468/478 landmarks) including detailed
    forehead and cheek contours for precision rPPG ROI extraction.
    Supports modern MediaPipe Tasks API, legacy solutions.face_mesh fallback,
    and automatic YuNetTracker fallback if MediaPipe is unavailable.
    """

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )

    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str | Path | None = None,
    ) -> None:
        """Initialize MediaPipe Face Mesh tracker.

        Args:
            max_num_faces: Maximum number of faces to detect (default: 1).
            refine_landmarks: Refine landmarks around eyes/lips/irises (default: True).
            min_detection_confidence: Detection confidence threshold (default: 0.5).
            min_tracking_confidence: Tracking confidence threshold (default: 0.5).
            model_path: Optional custom path to face_landmarker.task file.
        """
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._backend: str = "none"
        self._fallback_tracker: Optional[BaseTracker] = None
        self._landmarker = None
        self._face_mesh = None
        self._mp = None

        # Ensure Windows DLL search directories are populated for mediapipe if needed
        if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
            py_dir = Path(sys.executable).parent
            if py_dir.is_dir():
                try:
                    os.add_dll_directory(str(py_dir))
                except Exception:
                    pass

        # 1. Try modern MediaPipe Tasks API
        try:
            import mediapipe as mp

            self._mp = mp

            tasks_vision = None
            if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
                tasks_vision = mp.tasks.vision
            elif hasattr(mp, "tasks") and hasattr(mp.tasks, "python") and hasattr(mp.tasks.python, "vision"):
                tasks_vision = mp.tasks.python.vision

            if tasks_vision is not None and hasattr(tasks_vision, "FaceLandmarker"):
                resolved_model_path = model_path or self._ensure_model_downloaded()
                model_path_str = str(resolved_model_path)
                if not os.path.exists(model_path_str):
                    raise FileNotFoundError(f"MediaPipe model not found at {model_path_str}")

                BaseOptions = getattr(mp.tasks, "BaseOptions", None)
                if BaseOptions is None and hasattr(mp.tasks, "python"):
                    BaseOptions = getattr(mp.tasks.python, "BaseOptions", None)
                if BaseOptions is None and hasattr(mp.tasks, "python") and hasattr(mp.tasks.python, "core"):
                    base_options_mod = getattr(mp.tasks.python.core, "base_options", None)
                    if base_options_mod is not None:
                        BaseOptions = getattr(base_options_mod, "BaseOptions", None)

                if BaseOptions is None:
                    raise ImportError("BaseOptions not found in mediapipe.tasks")

                base_opts = BaseOptions(model_asset_path=model_path_str)
                options = tasks_vision.FaceLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=tasks_vision.RunningMode.IMAGE,
                    num_faces=max_num_faces,
                    min_face_detection_confidence=min_detection_confidence,
                    min_face_presence_confidence=min_tracking_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self._landmarker = tasks_vision.FaceLandmarker.create_from_options(options)
                self._backend = "tasks"
                return
        except Exception:
            pass

        # 2. Try legacy mp.solutions.face_mesh
        try:
            import mediapipe as mp

            self._mp = mp

            solutions = getattr(mp, "solutions", None)
            if solutions is None:
                from mediapipe.python import solutions as mp_solutions  # type: ignore[import-not-found]

                solutions = mp_solutions

            face_mesh_module = getattr(solutions, "face_mesh", None)
            if face_mesh_module is not None:
                self._face_mesh = face_mesh_module.FaceMesh(
                    max_num_faces=max_num_faces,
                    refine_landmarks=refine_landmarks,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self._backend = "solutions"
                return
        except Exception:
            pass

        # 3. Gracefully fall back to YuNetTracker if MediaPipe is not available
        try:
            self._fallback_tracker = YuNetTracker(conf_threshold=min_detection_confidence)
            self._backend = "yunet"
        except Exception as e:
            raise ImportError(
                "Neither MediaPipe nor YuNetTracker could be initialized. "
                "Ensure mediapipe or opencv-python is properly installed."
            ) from e

    @classmethod
    def _ensure_model_downloaded(cls) -> Path:
        """Download MediaPipe face landmarker model into ~/.cache/rppg/models if missing."""
        cache_dir = Path.home() / ".cache" / "rppg" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_file = cache_dir / "face_landmarker.task"

        if not model_file.is_file() or model_file.stat().st_size == 0:
            try:
                req = urllib.request.Request(
                    cls.MODEL_URL,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
                )
                with urllib.request.urlopen(req) as resp, open(model_file, "wb") as out_file:
                    out_file.write(resp.read())
            except Exception as e:
                raise RuntimeError(
                    f"Failed to automatically download MediaPipe face landmarker model from {cls.MODEL_URL}: {e}. "
                    "Please provide `model_path` explicitly."
                ) from e

        return model_file

    def process(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarksResult]:
        """Detect facial mesh landmarks in a BGR frame.

        Args:
            frame_bgr: Video frame in BGR format of shape (H, W, 3).

        Returns:
            FaceLandmarksResult if face detected, None otherwise.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        if self._backend == "yunet" and self._fallback_tracker is not None:
            return self._fallback_tracker.process(frame_bgr)

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self._backend == "tasks" and self._landmarker is not None and self._mp is not None:
            mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            results = self._landmarker.detect(mp_img)
            if not results.face_landmarks:
                return None
            mesh = results.face_landmarks[0]
            pts_norm = np.array(
                [[lm.x, lm.y] for lm in mesh],
                dtype=np.float32,
            )
        elif self._backend == "solutions" and self._face_mesh is not None:
            frame_rgb.flags.writeable = False
            results = self._face_mesh.process(frame_rgb)
            if not results.multi_face_landmarks:
                return None
            mesh = results.multi_face_landmarks[0]
            pts_norm = np.array(
                [[lm.x, lm.y] for lm in mesh.landmark],
                dtype=np.float32,
            )
        else:
            return None

        pts_px = np.column_stack([pts_norm[:, 0] * w, pts_norm[:, 1] * h]).astype(np.float32)

        x_min = int(np.clip(np.min(pts_px[:, 0]), 0, w))
        y_min = int(np.clip(np.min(pts_px[:, 1]), 0, h))
        x_max = int(np.clip(np.max(pts_px[:, 0]), 0, w))
        y_max = int(np.clip(np.max(pts_px[:, 1]), 0, h))
        bbox = (x_min, y_min, max(0, x_max - x_min), max(0, y_max - y_min))

        return FaceLandmarksResult(
            bbox=bbox,
            landmarks=pts_px,
            normalized_landmarks=pts_norm,
            confidence=1.0,
        )

    def close(self) -> None:
        """Close tracker session and release resources."""
        if self._backend == "tasks" and self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        elif self._backend == "solutions" and self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        elif self._backend == "yunet" and self._fallback_tracker is not None:
            self._fallback_tracker.close()
            self._fallback_tracker = None


class YuNetTracker(BaseTracker):
    """Ultra-lightweight face detector and tracker using OpenCV FaceDetectorYN.

    Includes automatic download of the ONNX model from the official OpenCV Zoo
    if not present locally.
    """

    MODEL_URL = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )

    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        """Initialize YuNet tracker.

        Args:
            model_path: Path to face_detection_yunet.onnx file. If None, downloads to ~/.cache/rppg/.
            conf_threshold: Minimum confidence score threshold (default: 0.6).
            nms_threshold: Non-maximum suppression threshold (default: 0.3).
            top_k: Maximum number of faces to keep before NMS (default: 5000).
        """
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("cv2.FaceDetectorYN is not available in your OpenCV installation.")

        if model_path is None:
            model_path = self._ensure_model_downloaded()

        model_path_str = str(model_path)
        if not os.path.exists(model_path_str):
            raise FileNotFoundError(f"YuNet ONNX model not found at {model_path_str}")

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._input_size = (320, 320)

        self._detector = cv2.FaceDetectorYN.create(
            model=model_path_str,
            config="",
            input_size=self._input_size,
            score_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
            target_id=cv2.dnn.DNN_TARGET_CPU,
        )

    @classmethod
    def _ensure_model_downloaded(cls) -> Path:
        """Download YuNet ONNX model into ~/.cache/rppg if missing."""
        cache_dir = Path.home() / ".cache" / "rppg"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_file = cache_dir / "face_detection_yunet_2023mar.onnx"

        if not model_file.is_file() or model_file.stat().st_size == 0:
            try:
                urllib.request.urlretrieve(cls.MODEL_URL, str(model_file))
            except Exception as e:
                raise RuntimeError(
                    f"Failed to automatically download YuNet model from {cls.MODEL_URL}: {e}. "
                    "Please provide `model_path` explicitly."
                ) from e

        return model_file

    def process(self, frame_bgr: np.ndarray) -> Optional[FaceLandmarksResult]:
        """Detect face and keypoints using YuNet.

        Args:
            frame_bgr: Video frame in BGR format of shape (H, W, 3).

        Returns:
            FaceLandmarksResult if face detected, None otherwise.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]
        if self._input_size != (w, h):
            self._input_size = (w, h)
            self._detector.setInputSize((w, h))

        _, faces = self._detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None

        # Select highest-confidence face
        best_idx = int(np.argmax(faces[:, 14]))
        face = faces[best_idx]

        bx, by, bw, bh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
        confidence = float(face[14])

        # 5 keypoints: right eye, left eye, nose, right mouth corner, left mouth corner
        pts_px = np.array(
            [
                [face[4], face[5]],
                [face[6], face[7]],
                [face[8], face[9]],
                [face[10], face[11]],
                [face[12], face[13]],
            ],
            dtype=np.float32,
        )

        pts_norm = np.column_stack(
            [pts_px[:, 0] / max(w, 1), pts_px[:, 1] / max(h, 1)]
        ).astype(np.float32)

        return FaceLandmarksResult(
            bbox=(bx, by, bw, bh),
            landmarks=pts_px,
            normalized_landmarks=pts_norm,
            confidence=confidence,
        )
