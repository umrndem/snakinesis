from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from snakinesis.app import _camera_error_lines, _face_loss_notice, _face_loss_status
from snakinesis.gesture import GazeDirection
from snakinesis.landmarks import (
    CHIN,
    FACE_LEFT_EDGE,
    FACE_RIGHT_EDGE,
    FOREHEAD,
    LEFT_EYE_EAR,
    NOSE_TIP,
    RIGHT_EYE_EAR,
)
from snakinesis.snake_game import BonusFood, BoundaryMode, HeadGestureController, SnakeGame, SnakeScreen
from snakinesis.snake_game.game import APP_VERSION, AUTHORS
from snakinesis.sound import SoundPlayer
from snakinesis.tracker import FaceTracker


@dataclass
class FakeTracking:
    head_h_ratio: float = 0.50
    head_v_ratio: float = 0.50
    head_roll_deg: float = 0.0


class FaceLossNoticeTests(unittest.TestCase):
    def test_dark_obstruction_reports_face_covered(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)

        self.assertEqual(_face_loss_notice(frame), "Face covered - game paused")

    def test_skin_colored_obstruction_reports_face_covered(self) -> None:
        frame = np.full((120, 160, 3), (90, 140, 200), dtype=np.uint8)

        self.assertEqual(_face_loss_notice(frame), "Face covered - game paused")

    def test_partial_hand_obstruction_reports_face_covered(self) -> None:
        frame = np.full((120, 160, 3), (180, 120, 40), dtype=np.uint8)
        frame[44:76, 60:100] = (45, 70, 110)

        self.assertEqual(_face_loss_notice(frame), "Face covered - game paused")
        self.assertEqual(_face_loss_status(frame), "Face covered")

    def test_visible_background_reports_face_out_of_frame(self) -> None:
        frame = np.full((120, 160, 3), (180, 120, 40), dtype=np.uint8)

        self.assertEqual(_face_loss_notice(frame), "Face out of frame - game paused")
        self.assertEqual(_face_loss_status(frame), "Face out of frame")

    def test_purple_background_reports_face_out_of_frame(self) -> None:
        frame = np.full((120, 160, 3), (90, 60, 120), dtype=np.uint8)

        self.assertEqual(_face_loss_notice(frame), "Face out of frame - game paused")


class AppPromptTests(unittest.TestCase):
    def test_camera_error_lines_include_camera_index(self) -> None:
        lines = _camera_error_lines(3)

        self.assertIn("camera index 3", lines[0])
        self.assertTrue(any("privacy permissions" in line for line in lines))


def make_controller(calibration_frames: int = 4) -> HeadGestureController:
    return HeadGestureController(
        calibration_frames=calibration_frames,
        activation_threshold=0.107,
        release_threshold=0.107,
        axis_bias=1.20,
        min_command_interval_s=0.14,
        neutral_rearm_s=0.10,
    )


def make_landmarks(*, center_x: float = 0.50, center_y: float = 0.50):
    landmarks = [SimpleNamespace(x=0.50, y=0.50) for _ in range(468)]
    points = {
        LEFT_EYE_EAR[0]: (center_x - 0.20, center_y),
        LEFT_EYE_EAR[1]: (center_x - 0.16, center_y - 0.03),
        LEFT_EYE_EAR[2]: (center_x - 0.12, center_y - 0.03),
        LEFT_EYE_EAR[3]: (center_x - 0.08, center_y),
        LEFT_EYE_EAR[4]: (center_x - 0.12, center_y + 0.03),
        LEFT_EYE_EAR[5]: (center_x - 0.16, center_y + 0.03),
        RIGHT_EYE_EAR[0]: (center_x + 0.08, center_y),
        RIGHT_EYE_EAR[1]: (center_x + 0.12, center_y - 0.03),
        RIGHT_EYE_EAR[2]: (center_x + 0.16, center_y - 0.03),
        RIGHT_EYE_EAR[3]: (center_x + 0.20, center_y),
        RIGHT_EYE_EAR[4]: (center_x + 0.16, center_y + 0.03),
        RIGHT_EYE_EAR[5]: (center_x + 0.12, center_y + 0.03),
        NOSE_TIP: (center_x, center_y),
        FACE_LEFT_EDGE: (center_x - 0.26, center_y),
        FACE_RIGHT_EDGE: (center_x + 0.26, center_y),
        FOREHEAD: (center_x, center_y - 0.22),
        CHIN: (center_x, center_y + 0.24),
    }
    for idx, (x, y) in points.items():
        landmarks[idx] = SimpleNamespace(x=x, y=y)
    return landmarks


class FaceCenterTrackerTests(unittest.TestCase):
    def test_moving_face_up_lowers_head_vertical_ratio(self) -> None:
        center_tracker = FaceTracker(head_smoothing_alpha=1.0)
        up_tracker = FaceTracker(head_smoothing_alpha=1.0)

        center = center_tracker.compute(landmarks=make_landmarks(center_y=0.50), frame_shape=(480, 640, 3), head_motion_scale=1.0)
        up = up_tracker.compute(landmarks=make_landmarks(center_y=0.35), frame_shape=(480, 640, 3), head_motion_scale=1.0)

        self.assertIsNotNone(center)
        self.assertIsNotNone(up)
        self.assertLess(up.head_v_ratio, center.head_v_ratio)

    def test_moving_face_right_raises_head_horizontal_ratio(self) -> None:
        center_tracker = FaceTracker(head_smoothing_alpha=1.0)
        right_tracker = FaceTracker(head_smoothing_alpha=1.0)

        center = center_tracker.compute(landmarks=make_landmarks(center_x=0.50), frame_shape=(480, 640, 3), head_motion_scale=1.0)
        right = right_tracker.compute(landmarks=make_landmarks(center_x=0.65), frame_shape=(480, 640, 3), head_motion_scale=1.0)

        self.assertIsNotNone(center)
        self.assertIsNotNone(right)
        self.assertGreater(right.head_h_ratio, center.head_h_ratio)


class HeadGestureControllerTests(unittest.TestCase):
    def test_calibrates_from_neutral_samples(self) -> None:
        controller = make_controller()
        signal = None
        for frame in range(4):
            signal = controller.update(FakeTracking(), face_present=True, now_s=frame * 0.03)

        self.assertIsNotNone(signal)
        self.assertTrue(signal.calibrated)
        self.assertAlmostEqual(signal.calibration_progress, 1.0)

    def test_detects_direction_relative_to_calibrated_baseline(self) -> None:
        controller = make_controller()
        for frame in range(4):
            controller.update(FakeTracking(head_h_ratio=0.42), face_present=True, now_s=frame * 0.03)

        signal = controller.update(FakeTracking(head_h_ratio=0.55), face_present=True, now_s=1.0)

        self.assertEqual(signal.direction, GazeDirection.RIGHT)
        self.assertGreater(signal.dx, 0.107)

    def test_ignores_small_head_drift(self) -> None:
        controller = make_controller()
        for frame in range(4):
            controller.update(FakeTracking(), face_present=True, now_s=frame * 0.03)

        signal = controller.update(FakeTracking(head_h_ratio=0.56), face_present=True, now_s=1.0)

        self.assertEqual(signal.direction, GazeDirection.CENTER)
        self.assertLess(signal.dx, 0.107)

    def test_detects_up_head_movement(self) -> None:
        controller = make_controller()
        for frame in range(4):
            controller.update(FakeTracking(), face_present=True, now_s=frame * 0.03)

        signal = controller.update(FakeTracking(head_v_ratio=0.39), face_present=True, now_s=1.0)

        self.assertEqual(signal.direction, GazeDirection.UP)
        self.assertLess(signal.dy, -0.107)

    def test_detects_matching_down_head_movement(self) -> None:
        controller = make_controller()
        for frame in range(4):
            controller.update(FakeTracking(), face_present=True, now_s=frame * 0.03)

        signal = controller.update(FakeTracking(head_v_ratio=0.61), face_present=True, now_s=1.0)

        self.assertEqual(signal.direction, GazeDirection.DOWN)
        self.assertGreater(signal.dy, 0.107)

    def test_detects_matching_left_head_movement(self) -> None:
        controller = make_controller()
        for frame in range(4):
            controller.update(FakeTracking(), face_present=True, now_s=frame * 0.03)

        signal = controller.update(FakeTracking(head_h_ratio=0.39), face_present=True, now_s=1.0)

        self.assertEqual(signal.direction, GazeDirection.LEFT)
        self.assertLess(signal.dx, -0.107)

    def test_head_gesture_only_fires_once_until_neutral(self) -> None:
        controller = make_controller()
        for frame in range(4):
            controller.update(FakeTracking(), face_present=True, now_s=frame * 0.03)

        first = controller.update(FakeTracking(head_h_ratio=0.62), face_present=True, now_s=1.00)
        blocked = controller.update(FakeTracking(head_v_ratio=0.38), face_present=True, now_s=1.25)
        neutral_start = controller.update(FakeTracking(head_h_ratio=0.56), face_present=True, now_s=1.35)
        neutral_ready = controller.update(FakeTracking(head_h_ratio=0.56), face_present=True, now_s=1.50)
        second = controller.update(FakeTracking(head_v_ratio=0.38), face_present=True, now_s=1.75)

        self.assertEqual(first.direction, GazeDirection.RIGHT)
        self.assertEqual(blocked.direction, GazeDirection.CENTER)
        self.assertEqual(neutral_start.direction, GazeDirection.CENTER)
        self.assertEqual(neutral_ready.direction, GazeDirection.CENTER)
        self.assertEqual(second.direction, GazeDirection.UP)

class SnakeGameTests(unittest.TestCase):
    def test_snake_respects_step_timing(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)
        start_head = game._snake[0]

        game.update(1.0)
        first_head = game._snake[0]
        game.update(1.2)
        second_head = game._snake[0]

        self.assertNotEqual(start_head, first_head)
        self.assertEqual(first_head, second_head)

    def test_snake_moves_half_grid_unit_per_step(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)
        start_head = game._snake[0]

        game.update(1.0)

        self.assertEqual(game._snake[0], (start_head[0] + 1, start_head[1]))

    def test_queued_turn_waits_until_grid_alignment(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)
        start_head = game._snake[0]

        game.update(1.0)
        game.set_direction(GazeDirection.UP)
        game.update(1.6)
        aligned_head = game._snake[0]
        game.update(2.2)

        self.assertEqual(aligned_head, (start_head[0] + 2, start_head[1]))
        self.assertEqual(game._snake[0], (start_head[0] + 2, start_head[1] - 1))

    def test_render_letterboxes_wide_window_without_stretching(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)

        rendered = game.render(
            tracking_available=True,
            control_direction=GazeDirection.CENTER,
            control_status="Ready",
            calibration_progress=1.0,
            dx=0.0,
            dy=0.0,
            activation_radius=0.095,
            release_radius=0.095,
            target_size=(900, 600),
        )

        self.assertEqual(rendered.shape, (600, 900, 3))
        self.assertTrue((rendered[:, 0] == 0).all())
        self.assertTrue((rendered[:, -1] == 0).all())

    def test_wall_collision_wraps_to_other_side(self) -> None:
        game = SnakeGame(5, 5, 10, 0.50)
        game._snake = deque([(9, 4), (8, 4), (7, 4)])
        game._direction = GazeDirection.RIGHT
        game._pending_direction = GazeDirection.RIGHT
        game._food = (1, 1)

        game.update(1.0)

        self.assertEqual(game._snake[0], (0, 4))
        self.assertFalse(game.game_over)

    def test_wall_mode_ends_game_at_boundary(self) -> None:
        game = SnakeGame(5, 5, 10, 0.50)
        game._boundary_mode = BoundaryMode.WALLS
        game._snake = deque([(7, 4), (6, 4), (5, 4)])
        game._direction = GazeDirection.RIGHT
        game._pending_direction = GazeDirection.RIGHT
        game._food = (1, 1)

        game.update(1.0)

        self.assertTrue(game.game_over)
        self.assertEqual(game.screen, SnakeScreen.GAME_OVER)

    def test_wall_mode_food_spawns_inside_wall_ring(self) -> None:
        game = SnakeGame(8, 8, 10, 0.50)
        game._boundary_mode = BoundaryMode.WALLS

        for _ in range(25):
            x, y = game._spawn_food()
            self.assertGreater(x, 0)
            self.assertGreater(y, 0)
            self.assertLess(x, game.grid_width - 1)
            self.assertLess(y, game.grid_height - 1)

    def test_bonus_food_scores_big_reward(self) -> None:
        game = SnakeGame(6, 6, 10, 0.50, bonus_food_score=5)
        game._snake = deque([(2, 2), (1, 2), (0, 2)])
        game._direction = GazeDirection.RIGHT
        game._pending_direction = GazeDirection.RIGHT
        game._food = (5, 5)
        game._bonus_food = BonusFood(cells=((2, 1), (3, 1), (2, 2), (3, 2)), expires_at_s=10.0)
        starting_length = len(game._snake)

        game.update(1.0)
        game.update(1.6)

        self.assertEqual(game.score, 5)
        self.assertIsNone(game._bonus_food)
        self.assertEqual(len(game._snake), starting_length)
        self.assertEqual(game.pop_sound_events(), ("bonus_food_pickup",))

    def test_menu_right_sector_selects_after_hold(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True, menu_hold_s=1.25, menu_side_threshold=0.107)

        game.update_menu_control(direction=GazeDirection.CENTER, dx=0.13, dy=0.10, roll_delta_deg=0.0, now_s=0.0)
        game.update_menu_control(direction=GazeDirection.CENTER, dx=0.13, dy=0.10, roll_delta_deg=0.0, now_s=0.8)
        self.assertEqual(game.screen, SnakeScreen.MAIN_MENU)
        game.update_menu_control(direction=GazeDirection.CENTER, dx=0.13, dy=0.10, roll_delta_deg=0.0, now_s=1.3)

        self.assertEqual(game.screen, SnakeScreen.PLAYING)

    def test_main_menu_toggles_classic_walls_with_right_sector(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True, menu_hold_s=1.25, menu_side_threshold=0.107)

        game._menu_index = 1
        game.update_menu_control(direction=GazeDirection.CENTER, dx=0.13, dy=-0.06, roll_delta_deg=0.0, now_s=0.0)
        game.update_menu_control(direction=GazeDirection.CENTER, dx=0.13, dy=-0.06, roll_delta_deg=0.0, now_s=1.3)

        self.assertEqual(game.boundary_mode, BoundaryMode.WALLS)
        self.assertEqual(game.screen, SnakeScreen.MAIN_MENU)

    def test_main_menu_toggles_sound(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True)

        game._menu_index = 2
        game._select_menu_item()

        self.assertFalse(game.sound_enabled)
        self.assertIn("Sound: OFF", game._main_menu_items)
        self.assertEqual(game.pop_sound_events(), ("menu_select",))

    def test_main_menu_toggles_camera_flip(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True, camera_flipped=True)

        game._menu_index = 3
        game._select_menu_item()

        self.assertFalse(game.camera_flipped)
        self.assertIn("Camera Flip: OFF", game._main_menu_items)
        self.assertEqual(game.pop_sound_events(), ("menu_select",))

    def test_pause_menu_includes_sound_toggle(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)

        game.open_pause_menu()

        self.assertIn("Sound: ON", game._main_menu_items)
        self.assertIn("Camera Flip: ON", game._main_menu_items)
        self.assertNotIn("Classic Walls: OFF", game._main_menu_items)
        self.assertEqual(game.pop_sound_events(), ("pause",))

    def test_enabling_sound_queues_select_sound_after_toggle(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True)
        game._sound_enabled = False

        game._menu_index = 2
        game._select_menu_item()

        self.assertTrue(game.sound_enabled)
        self.assertEqual(game.pop_sound_events(), ("menu_select",))

    def test_menu_browsing_queues_sound_event(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True)

        game.handle_key(ord("s"))

        self.assertEqual(game.pop_sound_events(), ("menu_move",))
        self.assertEqual(game.pop_sound_events(), ())

    def test_disabled_sound_suppresses_sound_events(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True)

        game._toggle_sound()
        game.handle_key(ord("s"))

        self.assertEqual(game.pop_sound_events(), ())

    def test_main_menu_opens_instructions(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True)

        game._menu_index = 4
        game._select_menu_item()

        self.assertEqual(game.screen, SnakeScreen.INSTRUCTIONS)
        self.assertEqual(game.pop_sound_events(), ("menu_select",))

    def test_main_menu_opens_about_with_authors_and_version(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True)

        game._menu_index = 6
        game._select_menu_item()

        self.assertEqual(game.screen, SnakeScreen.ABOUT)
        self.assertEqual(APP_VERSION, "1.1")
        self.assertIn(("Muhammad Umar Nadeem", "github.com/umrndem"), AUTHORS)
        self.assertIn(("Shifa Zeeshan", "github.com/AshwaZeeshan"), AUTHORS)
        self.assertEqual(game.pop_sound_events(), ("menu_select",))

    def test_pause_menu_hides_boundary_toggle(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)

        game.handle_key(27)

        self.assertEqual(game.screen, SnakeScreen.MAIN_MENU)
        self.assertNotIn("Classic Walls: OFF", game._main_menu_items)
        self.assertEqual(
            game._main_menu_items,
            ("Resume Game", "Sound: ON", "Camera Flip: ON", "Instructions", "High Scores", "About", "Return to Main Menu", "Quit"),
        )

    def test_pause_menu_resume_does_not_toggle_boundary_mode(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)

        game.handle_key(27)
        game._menu_index = 3
        game._select_menu_item()

        self.assertEqual(game.boundary_mode, BoundaryMode.BOUNDARYLESS)
        self.assertEqual(game.screen, SnakeScreen.INSTRUCTIONS)

    def test_pause_menu_return_to_main_menu_abandons_resume_option(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)

        game.open_pause_menu()
        game._menu_index = 6
        self.assertEqual(game._selected_menu_warning(), "ALL PROGRESS WILL BE LOST")
        game._select_menu_item()

        self.assertEqual(game.screen, SnakeScreen.MAIN_MENU)
        self.assertFalse(game._in_game_menu)
        self.assertIn("Start Game", game._main_menu_items)
        self.assertNotIn("Resume Game", game._main_menu_items)

    def test_face_cover_pause_menu_shows_notice(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)

        game.open_pause_menu("Face covered - game paused")

        self.assertEqual(game.screen, SnakeScreen.MAIN_MENU)
        self.assertTrue(game._in_game_menu)
        self.assertEqual(game._pause_notice, "Face covered - game paused")

    def test_quit_menu_shows_goodbye_before_closing(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True, exit_message_s=2.0)

        game._menu_index = len(game._main_menu_items) - 1
        game._select_menu_item()
        game.update(10.0)
        game.update(11.9)

        self.assertEqual(game.screen, SnakeScreen.EXITING)
        self.assertFalse(game.quit_requested)

        game.update(12.1)

        self.assertTrue(game.quit_requested)

    def test_q_key_starts_goodbye_screen(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)

        handled = game.handle_key(ord("q"))

        self.assertTrue(handled)
        self.assertEqual(game.screen, SnakeScreen.EXITING)
        self.assertFalse(game.quit_requested)
        self.assertEqual(game.pop_sound_events(), ("quit_hiss",))

    def test_back_from_submenu_queues_back_sound(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True)
        game._screen = SnakeScreen.INSTRUCTIONS

        game.handle_key(27)

        self.assertEqual(game.screen, SnakeScreen.MAIN_MENU)
        self.assertEqual(game.pop_sound_events(), ("menu_back",))

    def test_back_from_pause_menu_queues_back_sound(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)

        game.open_pause_menu()
        game.pop_sound_events()
        game.handle_key(27)

        self.assertEqual(game.screen, SnakeScreen.PLAYING)
        self.assertEqual(game.pop_sound_events(), ("menu_back",))

    def test_food_pickup_queues_sound_event(self) -> None:
        game = SnakeGame(6, 6, 10, 0.50)
        game._snake = deque([(2, 2), (1, 2), (0, 2)])
        game._direction = GazeDirection.RIGHT
        game._pending_direction = GazeDirection.RIGHT
        game._food = (2, 1)

        game.update(1.0)
        game.update(1.6)

        self.assertEqual(game.pop_sound_events(), ("food_pickup",))

    def test_death_queues_sound_event(self) -> None:
        game = SnakeGame(5, 5, 10, 0.50)
        game._boundary_mode = BoundaryMode.WALLS
        game._snake = deque([(7, 4), (6, 4), (5, 4)])
        game._direction = GazeDirection.RIGHT
        game._pending_direction = GazeDirection.RIGHT
        game._food = (1, 1)

        game.update(1.0)

        self.assertTrue(game.game_over)
        self.assertEqual(game.pop_sound_events(), ("death",))

    def test_sound_player_maps_all_game_feedback_events(self) -> None:
        expected = {
            "menu_move",
            "menu_select",
            "menu_back",
            "pause",
            "food_pickup",
            "bonus_food_pickup",
            "death",
            "quit_hiss",
        }
        player = SoundPlayer()

        self.assertTrue(expected.issubset(player._sounds.keys()))
        for sound_name in expected:
            self.assertTrue(player._sounds[sound_name].exists(), sound_name)

    def test_hold_progress_uses_top_global_bar_without_bottom_text(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)
        board = np.zeros((500, 600, 3), dtype=np.uint8)
        game._menu_hold_action = "back"
        game._menu_hold_progress = 0.50

        game._draw_hold_progress(board, 600, 500)

        self.assertTrue(board[6:29].any())
        self.assertFalse(board[34:70].any())
        self.assertFalse(board[360:470].any())

    def test_right_sector_does_nothing_on_info_panes(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True, menu_hold_s=1.25, menu_side_threshold=0.107)

        for screen in (SnakeScreen.INSTRUCTIONS, SnakeScreen.HIGH_SCORES, SnakeScreen.ABOUT):
            game._screen = screen
            game.update_menu_control(direction=GazeDirection.CENTER, dx=0.13, dy=0.02, roll_delta_deg=0.0, now_s=0.0)
            game.update_menu_control(direction=GazeDirection.CENTER, dx=0.13, dy=0.02, roll_delta_deg=0.0, now_s=1.3)

            self.assertEqual(game.screen, screen)
            self.assertIsNone(game._menu_hold_action)
            self.assertEqual(game._menu_hold_progress, 0.0)

    def test_left_sector_backs_out_of_high_scores(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50, start_in_menu=True, menu_hold_s=1.25, menu_side_threshold=0.107)

        game._screen = SnakeScreen.HIGH_SCORES
        game.update_menu_control(direction=GazeDirection.CENTER, dx=-0.13, dy=0.05, roll_delta_deg=0.0, now_s=0.0)
        game.update_menu_control(direction=GazeDirection.CENTER, dx=-0.13, dy=0.05, roll_delta_deg=0.0, now_s=1.3)

        self.assertEqual(game.screen, SnakeScreen.MAIN_MENU)

    def test_escape_returns_to_main_menu_during_gameplay(self) -> None:
        game = SnakeGame(10, 10, 10, 0.50)

        handled = game.handle_key(27)

        self.assertTrue(handled)
        self.assertEqual(game.screen, SnakeScreen.MAIN_MENU)

    def test_high_score_persists_per_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}\\scores.json"
            game = SnakeGame(10, 10, 10, 0.50, high_score_path=path)
            game._score = 9
            game._finish_game()

            loaded = SnakeGame(10, 10, 10, 0.50, high_score_path=path)

        self.assertEqual(loaded.high_score, 9)


if __name__ == "__main__":
    unittest.main()
