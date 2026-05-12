from __future__ import annotations

# MediaPipe FaceMesh indices used by the Snake head tracker.

# Eye outline points used to estimate head roll.
LEFT_EYE_EAR = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_EAR = (362, 385, 387, 263, 373, 380)

# Stable face points used to estimate face-center movement.
NOSE_TIP = 1
FACE_LEFT_EDGE = 234
FACE_RIGHT_EDGE = 454
FOREHEAD = 10
CHIN = 152
