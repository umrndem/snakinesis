from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    target_fps: int = 30
    camera_width: int = 640
    camera_height: int = 480
    snake_window_name: str = "Snakinesis"
    snake_window_width: int = 720
    snake_window_height: int = 860

    # Snake game
    snake_grid_width: int = 16
    snake_grid_height: int = 16
    snake_cell_size: int = 36
    snake_movement_units_per_cell: int = 2
    snake_step_s: float = 0.22
    snake_high_score_file: str = "snake_high_scores.json"
    snake_bonus_food_every: int = 4
    snake_bonus_food_score: int = 5
    snake_bonus_food_duration_s: float = 6.0
    snake_exit_message_s: float = 2.4
    snake_menu_hold_s: float = 1.25
    snake_menu_tilt_threshold_deg: float = 9.0
    snake_menu_tilt_release_deg: float = 4.0
    snake_menu_side_threshold: float = 0.107
    snake_menu_side_release_threshold: float = 0.085
    snake_face_loss_pause_s: float = 0.45
    snake_head_motion_scale: float = 1.85
    snake_head_smoothing_alpha: float = 0.60
    snake_processing_scale: float = 0.55
    snake_calibration_frames: int = 18
    snake_control_threshold: float = 0.118
    snake_release_threshold: float = 0.118
    snake_axis_bias: float = 1.20
    snake_min_command_interval_s: float = 0.14
    snake_neutral_rearm_s: float = 0.10

    # Misc
    flip_selfie: bool = True
