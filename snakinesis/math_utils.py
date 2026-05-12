from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class ExponentialSmoother:
    alpha: float
    _value: Optional[float] = None

    def push(self, v: float) -> float:
        value = float(v)
        alpha = min(max(float(self.alpha), 0.0), 1.0)
        if self._value is None:
            self._value = value
        else:
            self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value


def mean_point(points: Iterable[np.ndarray]) -> np.ndarray:
    pts = np.array(list(points), dtype=np.float32)
    return pts.mean(axis=0)
