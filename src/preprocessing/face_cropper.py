
from typing import Tuple
import cv2
import numpy as np


class FaceCropper:
    """
    Crops and resizes face regions from raw video frames.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        
        self.target_size = target_size

    def crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        margin: float = 0.2
    ) -> np.ndarray:
        """
        Expands the bounding box by a given margin, crops the region,
        clips to frame boundaries, converts to RGB, and resizes.

        Args:
            frame: Input video frame array [H, W, 3] (assumed BGR from OpenCV).
            bbox: Bounding box tuple (x1, y1, x2, y2) in pixel coordinates.
            margin: Fractional expansion margin relative to width/height.

        Returns:
            RGB face crop array [target_size[0], target_size[1], 3].
        """
        if frame is None or frame.size == 0:
            raise ValueError("Invalid or empty frame provided to FaceCropper.")

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        # Calculate width and height of the raw detection box
        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bounding box dimensions: width={w}, height={h}")

        # Expand box coordinates with margin
        x1_prime = int(round(x1 - margin * w))
        y1_prime = int(round(y1 - margin * h))
        x2_prime = int(round(x2 + margin * w))
        y2_prime = int(round(y2 + margin * h))

        # Clip coordinates strictly within frame boundaries
        x1_clipped = max(0, min(frame_w, x1_prime))
        y1_clipped = max(0, min(frame_h, y1_prime))
        x2_clipped = max(0, min(frame_w, x2_prime))
        y2_clipped = max(0, min(frame_h, y2_prime))

        # Check if the clipped area is valid
        if x2_clipped <= x1_clipped or y2_clipped <= y1_clipped:
            raise ValueError("Clipped bounding box coordinates resulted in a zero-area crop.")

        # Extract slice from frame [H, W, C]
        face_crop = frame[y1_clipped:y2_clipped, x1_clipped:x2_clipped]

        # Convert BGR (OpenCV default) to RGB
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Resize to target dimensions (224x224) using bilinear/area interpolation
        interpolation = cv2.INTER_AREA if (face_crop.shape[0] > self.target_size[0]) else cv2.INTER_LINEAR
        resized_face = cv2.resize(face_crop_rgb, self.target_size, interpolation=interpolation)

        return resized_face