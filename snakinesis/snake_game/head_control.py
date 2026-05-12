from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from statistics import median
from typing import List, Optional

from ..gesture import GazeDirection


@dataclass(frozen=True)
class HeadGestureSignal:
    direction: GazeDirection
    calibrated: bool
    calibration_progress: float
    dx: float
    dy: float
    roll_delta_deg: float
    status: str


class HeadGestureController:
    def __init__(
        self,
        *,
        calibration_frames: int,
        activation_threshold: float,
        release_threshold: float,
        axis_bias: float,
        min_command_interval_s: float,
        neutral_rearm_s: float,
    ) -> None:
        self.calibration_frames = max(3, int(calibration_frames))
        self.activation_threshold = float(activation_threshold)
        self.release_threshold = float(release_threshold)
        self.axis_bias = max(float(axis_bias), 1.0)
        self.min_command_interval_s = float(min_command_interval_s)
        self.neutral_rearm_s = float(neutral_rearm_s)
        self.reset_calibration()

    def reset_calibration(self) -> None:
        self._samples: List[tuple[float, float, float]] = []
        self._base_head_h = 0.5
        self._base_head_v = 0.5
        self._base_roll_deg = 0.0
        self._last_direction = GazeDirection.CENTER
        self._last_command_time_s = -999.0
        self._neutral_since_s: Optional[float] = None
        self._armed = True
        self._calibrated = False

    def update(self, tracking, *, face_present: bool, now_s: float) -> HeadGestureSignal:
        if tracking is None or not face_present:
            return HeadGestureSignal(
                direction=GazeDirection.CENTER,
                calibrated=self._calibrated,
                calibration_progress=self.calibration_progress,
                dx=0.0,
                dy=0.0,
                roll_delta_deg=0.0,
                status="Searching for face",
            )

        if not self._calibrated:
            self._samples.append((tracking.head_h_ratio, tracking.head_v_ratio, getattr(tracking, "head_roll_deg", 0.0)))
            if len(self._samples) >= self.calibration_frames:
                self._finish_calibration()
            return HeadGestureSignal(
                direction=GazeDirection.CENTER,
                calibrated=self._calibrated,
                calibration_progress=self.calibration_progress,
                dx=0.0,
                dy=0.0,
                roll_delta_deg=0.0,
                status="Calibrating",
            )

        head_dx = tracking.head_h_ratio - self._base_head_h
        head_dy = tracking.head_v_ratio - self._base_head_v
        dx = head_dx
        dy = head_dy
        roll_delta_deg = getattr(tracking, "head_roll_deg", 0.0) - self._base_roll_deg
        direction = self._direction_from_delta(dx, dy, now_s)

        return HeadGestureSignal(
            direction=direction,
            calibrated=True,
            calibration_progress=1.0,
            dx=dx,
            dy=dy,
            roll_delta_deg=roll_delta_deg,
            status="Ready",
        )

    @property
    def calibration_progress(self) -> float:
        if self._calibrated:
            return 1.0
        return min(1.0, len(self._samples) / self.calibration_frames)

    def _finish_calibration(self) -> None:
        self._base_head_h = median(sample[0] for sample in self._samples)
        self._base_head_v = median(sample[1] for sample in self._samples)
        self._base_roll_deg = median(sample[2] for sample in self._samples)
        self._calibrated = True

    def _direction_from_delta(self, dx: float, dy: float, now_s: float) -> GazeDirection:
        magnitude = hypot(dx, dy)
        if magnitude <= self.release_threshold:
            if self._neutral_since_s is None:
                self._neutral_since_s = now_s
            if (now_s - self._neutral_since_s) >= self.neutral_rearm_s:
                self._armed = True
                self._last_direction = GazeDirection.CENTER
            return GazeDirection.CENTER

        self._neutral_since_s = None
        if not self._armed:
            self._last_direction = GazeDirection.CENTER
            return GazeDirection.CENTER

        if (now_s - self._last_command_time_s) < self.min_command_interval_s:
            return GazeDirection.CENTER

        abs_dx = abs(dx)
        abs_dy = abs(dy)

        if abs_dx >= abs_dy * self.axis_bias:
            direction = GazeDirection.RIGHT if dx > 0 else GazeDirection.LEFT
        elif abs_dy >= abs_dx * self.axis_bias:
            direction = GazeDirection.DOWN if dy > 0 else GazeDirection.UP
        else:
            if abs_dy > abs_dx:
                direction = GazeDirection.DOWN if dy > 0 else GazeDirection.UP
            else:
                direction = GazeDirection.RIGHT if dx > 0 else GazeDirection.LEFT

        if magnitude <= self.activation_threshold:
            return GazeDirection.CENTER

        if direction != self._last_direction:
            self._last_direction = direction
            self._last_command_time_s = now_s
            self._armed = False
            return direction
        return GazeDirection.CENTER
