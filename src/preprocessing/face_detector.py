
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

from urllib.request import urlretrieve

import numpy as np
import cv2

_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # DeepGuard/
_DEFAULT_MODEL = _PROJECT_ROOT / "model_downloaded" / "face_detection_yunet_2023mar.onnx"
_YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def _ensure_yunet_model(model_path: Path) -> Path:
    """Make sure the YuNet ONNX file exists locally, downloading it if needed."""
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(_YUNET_URL, model_path)

    if not model_path.exists() or model_path.stat().st_size == 0:
        raise FileNotFoundError(f"YuNet download failed: {model_path}")

    return model_path



@dataclass
class FaceDetection:
    """
    Standardized face detection container.
    Bbox format: (x1, y1, x2, y2) in integer pixel coordinates.
    """
    bbox: Tuple[int, ...]
    confidence: float
    landmarks: Optional[np.ndarray] = None


class FaceDetector(ABC):
    """Abstract Base Class for all face detection backends."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a single BGR/RGB frame.

        Args:
            frame: Image array [H, W, C] (uint8).

        Returns:
            List of FaceDetection instances.
        """
        pass

    @staticmethod
    def select_primary_face(
        detections: List[FaceDetection], 
        strategy: str = "combined"
    ) -> Optional[FaceDetection]:
        """
        Selects the primary face from a list of detections.

        Args:
            detections: List of FaceDetection objects found in the frame.
            strategy: Selection criterion ('area', 'confidence', or 'combined').

        Returns:
            The chosen FaceDetection instance, or None if the list is empty.
        """
        if not detections:
            return None

        if strategy == "area":
            # Select the face with the largest bounding box area (w * h)
            return max(
                detections,
                key=lambda d: max(0, d.bbox[2] - d.bbox[0]) * max(0, d.bbox[3] - d.bbox[1])
            )

        elif strategy == "confidence":
            # Select the face with the highest model detection score
            return max(detections, key=lambda d: d.confidence)

        elif strategy == "combined":
            # Rank by area weighted by confidence (Area * Score)
            return max(
                detections,
                key=lambda d: (
                    max(0, d.bbox[2] - d.bbox[0]) * 
                    max(0, d.bbox[3] - d.bbox[1]) * 
                    d.confidence
                )
            )

        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")



class OpenCVFaceDetector(FaceDetector):
   

    def __init__(self, conf_threshold: float = 0.6,
                 model_path: str | Path | None = None,
                 input_size: tuple[int, int] = (320, 320)):

        model_path = Path(model_path) if model_path else _DEFAULT_MODEL
        model_path = _ensure_yunet_model(model_path)

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            input_size,
            score_threshold=conf_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
            
        return [FaceDetection(
                bbox=(int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3])),
                confidence=float(f[-1])) for f in faces]