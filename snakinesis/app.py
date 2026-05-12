from __future__ import annotations

import ctypes
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp

from .config import AppConfig
from .gesture import GazeDirection
from .snake_game import HeadGestureController, SnakeGame
from .sound import SoundPlayer
from .tracker import FaceTracker

_WINDOW_ICON_HANDLES: list[int] = []


def _resource_path(*parts: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return base.joinpath(*parts)


def _window_is_open(window_name: str) -> bool:
    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def _window_target_size(window_name: str, fallback: Tuple[int, int]) -> Tuple[int, int]:
    try:
        _, _, width, height = cv2.getWindowImageRect(window_name)
    except (AttributeError, cv2.error):
        return fallback
    if width <= 0 or height <= 0:
        return fallback
    return (width, height)


def _center_crop(frame, margin_fraction: float):
    height, width = frame.shape[:2]
    margin_fraction = max(0.0, min(margin_fraction, 0.45))
    top = int(height * margin_fraction)
    bottom = max(top + 1, height - top)
    left = int(width * margin_fraction)
    right = max(left + 1, width - left)
    return frame[top:bottom, left:right]


def _skin_like_ratio(crop) -> float:
    if crop.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)
    b_channel, g_channel, r_channel = (channel.astype("int16") for channel in cv2.split(crop))
    warm_mask = (
        (r_channel >= 55)
        & (g_channel >= 35)
        & (r_channel >= b_channel + 10)
        & (g_channel >= b_channel - 24)
        & (r_channel >= g_channel - 28)
    )
    ycrcb_skin = (
        (y_channel >= 35)
        & (cr_channel >= 120)
        & (cr_channel <= 190)
        & (cb_channel >= 70)
        & (cb_channel <= 150)
    )
    hsv_skin = (
        (h_channel <= 25)
        & (s_channel >= 20)
        & (s_channel <= 235)
        & (v_channel >= 35)
    )
    return float(((ycrcb_skin | hsv_skin) & warm_mask).mean())


def _frame_likely_covered(frame) -> bool:
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        return False

    crop = _center_crop(frame, 0.20)
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 35.0:
        return True

    inner_crop = _center_crop(frame, 0.34)
    return _skin_like_ratio(crop) >= 0.18 or _skin_like_ratio(inner_crop) >= 0.28


def _face_loss_status(frame) -> str:
    if _frame_likely_covered(frame):
        return "Face covered"
    return "Face out of frame"


def _face_loss_notice(frame) -> str:
    return f"{_face_loss_status(frame)} - game paused"


def _configure_windows_app_identity() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Snakinesis.Game")
    except (AttributeError, OSError):
        pass


def _set_windows_window_icon(window_name: str) -> bool:
    if sys.platform != "win32":
        return True

    icon_path = _resource_path("assets", "snakinesis.ico")
    if not icon_path.exists():
        return False

    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, window_name)
    if not hwnd:
        return False

    image_icon = 1
    icon_small = 0
    icon_big = 1
    load_from_file = 0x00000010
    default_size = 0x00000040
    wm_seticon = 0x0080

    big_icon = user32.LoadImageW(None, str(icon_path), image_icon, 0, 0, load_from_file | default_size)
    small_icon = user32.LoadImageW(None, str(icon_path), image_icon, 16, 16, load_from_file)
    if big_icon:
        user32.SendMessageW(hwnd, wm_seticon, icon_big, big_icon)
        _WINDOW_ICON_HANDLES.append(big_icon)
    if small_icon:
        user32.SendMessageW(hwnd, wm_seticon, icon_small, small_icon)
        _WINDOW_ICON_HANDLES.append(small_icon)
    return bool(big_icon or small_icon)


def main() -> int:
    cfg = AppConfig()
    _configure_windows_app_identity()

    cap = cv2.VideoCapture(cfg.camera_index)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return 2

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)
    cap.set(cv2.CAP_PROP_FPS, cfg.target_fps)
    cv2.namedWindow(cfg.snake_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(cfg.snake_window_name, cfg.snake_window_width, cfg.snake_window_height)
    icon_applied = _set_windows_window_icon(cfg.snake_window_name)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    tracker = FaceTracker(head_smoothing_alpha=cfg.snake_head_smoothing_alpha)
    sound_player = SoundPlayer()
    snake_game = SnakeGame(
        grid_width=cfg.snake_grid_width,
        grid_height=cfg.snake_grid_height,
        cell_size=cfg.snake_cell_size,
        step_s=cfg.snake_step_s,
        movement_units_per_cell=cfg.snake_movement_units_per_cell,
        high_score_path=cfg.snake_high_score_file,
        bonus_food_every=cfg.snake_bonus_food_every,
        bonus_food_score=cfg.snake_bonus_food_score,
        bonus_food_duration_s=cfg.snake_bonus_food_duration_s,
        exit_message_s=cfg.snake_exit_message_s,
        menu_hold_s=cfg.snake_menu_hold_s,
        menu_tilt_threshold_deg=cfg.snake_menu_tilt_threshold_deg,
        menu_tilt_release_deg=cfg.snake_menu_tilt_release_deg,
        menu_side_threshold=cfg.snake_menu_side_threshold,
        menu_side_release_threshold=cfg.snake_menu_side_release_threshold,
        start_in_menu=True,
    )
    head_controller = HeadGestureController(
        calibration_frames=cfg.snake_calibration_frames,
        activation_threshold=cfg.snake_control_threshold,
        release_threshold=cfg.snake_release_threshold,
        axis_bias=cfg.snake_axis_bias,
        min_command_interval_s=cfg.snake_min_command_interval_s,
        neutral_rearm_s=cfg.snake_neutral_rearm_s,
    )
    face_missing_started_s: Optional[float] = None

    try:
        while True:
            if not _window_is_open(cfg.snake_window_name):
                break

            ok, frame = cap.read()
            if not ok:
                break

            if cfg.flip_selfie:
                frame = cv2.flip(frame, 1)

            tracking_frame = frame
            if cfg.snake_processing_scale < 0.99:
                tracking_frame = cv2.resize(
                    frame,
                    None,
                    fx=cfg.snake_processing_scale,
                    fy=cfg.snake_processing_scale,
                    interpolation=cv2.INTER_LINEAR,
                )

            rgb = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = face_mesh.process(rgb)
            rgb.flags.writeable = True

            tracking = None
            tracking_available = False
            if results.multi_face_landmarks:
                tracking_available = True
                landmarks = results.multi_face_landmarks[0].landmark
                tracking = tracker.compute(
                    landmarks=landmarks,
                    frame_shape=tracking_frame.shape,
                    head_motion_scale=cfg.snake_head_motion_scale,
                )

            now = time.perf_counter()
            signal = head_controller.update(tracking, face_present=tracking_available, now_s=now)
            face_loss_status = None if tracking_available else _face_loss_status(frame)
            if tracking_available:
                face_missing_started_s = None
            elif snake_game.accepts_direction_input:
                if face_missing_started_s is None:
                    face_missing_started_s = now
                elif (now - face_missing_started_s) >= cfg.snake_face_loss_pause_s:
                    snake_game.open_pause_menu(f"{face_loss_status} - game paused")
                    face_missing_started_s = None
            else:
                face_missing_started_s = None

            snake_game.update_menu_control(
                direction=signal.direction,
                dx=signal.dx,
                dy=signal.dy,
                roll_delta_deg=signal.roll_delta_deg,
                now_s=now,
            )
            if signal.direction != GazeDirection.CENTER:
                snake_game.set_direction(signal.direction)

            snake_game.update(now)
            for sound_event in snake_game.pop_sound_events():
                sound_player.play(sound_event)

            target_size = _window_target_size(
                cfg.snake_window_name,
                (cfg.snake_window_width, cfg.snake_window_height),
            )
            cv2.imshow(
                cfg.snake_window_name,
                snake_game.render(
                    tracking_available=tracking_available,
                    control_direction=signal.direction,
                    control_status=face_loss_status or signal.status,
                    calibration_progress=signal.calibration_progress,
                    dx=signal.dx,
                    dy=signal.dy,
                    activation_radius=cfg.snake_control_threshold,
                    release_radius=cfg.snake_release_threshold,
                    camera_preview=frame,
                    target_size=target_size,
                    now_s=now,
                ),
            )
            if not icon_applied:
                icon_applied = _set_windows_window_icon(cfg.snake_window_name)

            key = cv2.waitKey(1) & 0xFF
            handled = snake_game.handle_key(key)
            if key in (ord("c"), ord("C")):
                head_controller.reset_calibration()
            for sound_event in snake_game.pop_sound_events():
                sound_player.play(sound_event)
            if snake_game.quit_requested:
                break
            if key == 27 and not handled:
                break

    finally:
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()

    return 0
