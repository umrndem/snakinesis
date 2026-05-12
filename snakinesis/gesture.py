from __future__ import annotations

from enum import Enum


class GazeDirection(str, Enum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UP = "UP"
    DOWN = "DOWN"
    CENTER = "CENTER"
