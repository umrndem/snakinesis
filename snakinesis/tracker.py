from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Tuple

import numpy as np

from .landmarks import CHIN, FACE_LEFT_EDGE, FACE_RIGHT_EDGE, FOREHEAD, LEFT_EYE_EAR, NOSE_TIP, RIGHT_EYE_EAR
from .math_utils import ExponentialSmoother, clamp01, mean_point


@dataclass
class FaceTracking:
    head_h_ratio: float
    head_v_ratio: float
    head_roll_deg: float


class FaceTracker:
    def __init__(self, *, head_smoothing_alpha: float) -> None:
        self._head_h_smooth = ExponentialSmoother(head_smoothing_alpha)
        self._head_v_smooth = ExponentialSmoother(head_smoothing_alpha)

    @staticmethod
    def _lm_xy(landmarks, idx: int, width: int, height: int) -> np.ndarray:
        landmark = landmarks[idx]
        return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)

    def compute(self, *, landmarks, frame_shape: Tuple[int, int, int], head_motion_scale: float) -> FaceTracking:
        height, width = frame_shape[0], frame_shape[1]

        left_eye = tuple(self._lm_xy(landmarks, index, width, height) for index in LEFT_EYE_EAR)
        right_eye = tuple(self._lm_xy(landmarks, index, width, height) for index in RIGHT_EYE_EAR)
        left_eye_center = mean_point(left_eye)
        right_eye_center = mean_point(right_eye)
        eye_line = right_eye_center - left_eye_center
        head_roll_deg = degrees(atan2(float(eye_line[1]), float(eye_line[0])))

        head_points = [
            self._lm_xy(landmarks, index, width, height)
            for index in (NOSE_TIP, FACE_LEFT_EDGE, FACE_RIGHT_EDGE, FOREHEAD, CHIN)
        ]
        head_center = mean_point(head_points)
        scale = max(float(head_motion_scale), 0.1)
        head_h_ratio = clamp01(0.5 + ((float(head_center[0]) / max(width, 1)) - 0.5) * scale)
        head_v_ratio = clamp01(0.5 + ((float(head_center[1]) / max(height, 1)) - 0.5) * scale)

        return FaceTracking(
            head_h_ratio=self._head_h_smooth.push(head_h_ratio),
            head_v_ratio=self._head_v_smooth.push(head_v_ratio),
            head_roll_deg=head_roll_deg,
        )
