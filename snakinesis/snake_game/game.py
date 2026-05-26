from __future__ import annotations

import json
import random
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from math import hypot
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..gesture import GazeDirection

GridPoint = Tuple[int, int]
UnitPoint = Tuple[int, int]

APP_VERSION = "1.1.1"
AUTHORS = (
    ("Muhammad Umar Nadeem", "github.com/umrndem"),
    ("Shifa Zeeshan", "github.com/AshwaZeeshan"),
)


class BoundaryMode(str, Enum):
    BOUNDARYLESS = "Boundaryless"
    WALLS = "Classic Walls"


class SnakeScreen(str, Enum):
    MAIN_MENU = "MAIN_MENU"
    INSTRUCTIONS = "INSTRUCTIONS"
    HIGH_SCORES = "HIGH_SCORES"
    ABOUT = "ABOUT"
    PLAYING = "PLAYING"
    GAME_OVER = "GAME_OVER"
    EXITING = "EXITING"


@dataclass(frozen=True)
class BonusFood:
    cells: Tuple[GridPoint, ...]
    expires_at_s: float


def _resource_path(*parts: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[2]))
    return base.joinpath(*parts)


@lru_cache(maxsize=16)
def _pixel_font(size: int):
    font_path = _resource_path("assets", "fonts", "PressStart2P-Regular.ttf")
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


@lru_cache(maxsize=1)
def _logo_image() -> Optional[Image.Image]:
    logo_path = _resource_path("assets", "snakinesis_logo.png")
    if not logo_path.exists():
        return None
    return Image.open(logo_path).convert("RGBA")


@dataclass
class SnakeGame:
    grid_width: int
    grid_height: int
    cell_size: int
    step_s: float
    movement_units_per_cell: int = 2
    high_score_path: Optional[str] = None
    bonus_food_every: int = 4
    bonus_food_score: int = 5
    bonus_food_duration_s: float = 6.0
    exit_message_s: float = 2.4
    menu_hold_s: float = 1.25
    menu_tilt_threshold_deg: float = 9.0
    menu_tilt_release_deg: float = 4.0
    menu_side_threshold: float = 0.107
    menu_side_release_threshold: float = 0.085
    camera_flipped: bool = True
    start_in_menu: bool = False
    _snake: Deque[UnitPoint] = None  # type: ignore[assignment]
    _direction: GazeDirection = GazeDirection.RIGHT
    _pending_direction: GazeDirection = GazeDirection.RIGHT
    _food: GridPoint = (0, 0)
    _bonus_food: Optional[BonusFood] = None
    _score: int = 0
    _growth_units: int = 0
    _game_over: bool = False
    _paused: bool = False
    _last_step_time_s: float = 0.0
    _boundary_mode: BoundaryMode = BoundaryMode.BOUNDARYLESS
    _sound_enabled: bool = True
    _screen: SnakeScreen = SnakeScreen.PLAYING
    _menu_index: int = 0
    _game_over_index: int = 0
    _in_game_menu: bool = False
    _pause_notice: Optional[str] = None
    _normal_foods_since_bonus: int = 0
    _high_scores: dict = None  # type: ignore[assignment]
    _quit_requested: bool = False
    _exit_started_s: Optional[float] = None
    _sound_events: List[str] = None  # type: ignore[assignment]
    _menu_hold_action: Optional[str] = None
    _menu_hold_started_s: Optional[float] = None
    _menu_hold_progress: float = 0.0
    _menu_tilt_armed: bool = True

    def __post_init__(self) -> None:
        self.movement_units_per_cell = max(1, int(self.movement_units_per_cell))
        self.bonus_food_every = max(1, int(self.bonus_food_every))
        self.bonus_food_score = max(1, int(self.bonus_food_score))
        self.bonus_food_duration_s = max(1.0, float(self.bonus_food_duration_s))
        self.exit_message_s = max(1.0, float(self.exit_message_s))
        self.menu_hold_s = max(0.5, float(self.menu_hold_s))
        self.menu_tilt_threshold_deg = max(1.0, float(self.menu_tilt_threshold_deg))
        self.menu_tilt_release_deg = max(0.5, float(self.menu_tilt_release_deg))
        self.menu_side_threshold = max(0.01, float(self.menu_side_threshold))
        self.menu_side_release_threshold = min(self.menu_side_threshold, max(0.0, float(self.menu_side_release_threshold)))
        self.camera_flipped = bool(self.camera_flipped)
        self._high_scores = self._load_high_scores()
        self._sound_events = []
        self.reset()
        if self.start_in_menu:
            self._screen = SnakeScreen.MAIN_MENU

    def reset(self) -> None:
        cx = self.grid_width // 2
        cy = self.grid_height // 2
        head_x = cx * self.movement_units_per_cell
        head_y = cy * self.movement_units_per_cell
        snake_length = 3 * self.movement_units_per_cell
        self._snake = deque(((head_x - offset) % self._unit_width, head_y) for offset in range(snake_length))
        self._direction = GazeDirection.RIGHT
        self._pending_direction = GazeDirection.RIGHT
        self._score = 0
        self._growth_units = 0
        self._bonus_food = None
        self._normal_foods_since_bonus = 0
        self._game_over = False
        self._paused = False
        self._in_game_menu = False
        self._pause_notice = None
        self._quit_requested = False
        self._exit_started_s = None
        self._screen = SnakeScreen.PLAYING
        self._last_step_time_s = 0.0
        self._food = self._spawn_food()

    @property
    def score(self) -> int:
        return self._score

    @property
    def game_over(self) -> bool:
        return self._game_over

    @property
    def screen(self) -> SnakeScreen:
        return self._screen

    @property
    def boundary_mode(self) -> BoundaryMode:
        return self._boundary_mode

    @property
    def sound_enabled(self) -> bool:
        return self._sound_enabled

    @property
    def high_score(self) -> int:
        return int(self._high_scores.get(self._mode_key(self._boundary_mode), 0))

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def pop_sound_events(self) -> Tuple[str, ...]:
        events = tuple(self._sound_events)
        self._sound_events.clear()
        return events

    @property
    def accepts_direction_input(self) -> bool:
        return self._screen == SnakeScreen.PLAYING and not self._game_over and not self._paused

    def toggle_pause(self) -> None:
        if self._screen == SnakeScreen.PLAYING and not self._game_over:
            self.open_pause_menu()
        elif self._screen == SnakeScreen.MAIN_MENU and self._in_game_menu:
            self._resume_game()

    def open_pause_menu(self, notice: Optional[str] = None) -> None:
        if self._screen != SnakeScreen.PLAYING or self._game_over:
            return
        self._in_game_menu = True
        self._pause_notice = notice
        self._menu_index = 0
        self._screen = SnakeScreen.MAIN_MENU
        self._paused = False
        self._reset_menu_hold()
        self._queue_sound("pause")

    def request_quit(self) -> None:
        self._start_exit()

    def update_menu_control(self, *, direction: GazeDirection, dx: float, dy: float, roll_delta_deg: float, now_s: float) -> None:
        if self._screen == SnakeScreen.EXITING:
            self._reset_menu_hold()
            return
        if self._screen == SnakeScreen.PLAYING:
            self._reset_menu_hold()
            return

        if direction == GazeDirection.UP:
            self._move_menu(-1)
        elif direction == GazeDirection.DOWN:
            self._move_menu(1)

        if not self._menu_tilt_armed:
            if self._menu_action_released(dx=dx, dy=dy, roll_delta_deg=roll_delta_deg):
                self._menu_tilt_armed = True
            self._reset_menu_hold()
            return

        action = self._menu_action_from_pose(dx=dx, dy=dy, roll_delta_deg=roll_delta_deg)

        if action is None:
            self._reset_menu_hold()
            return

        if self._menu_hold_action != action:
            self._menu_hold_action = action
            self._menu_hold_started_s = now_s
            self._menu_hold_progress = 0.0
            return

        started_s = self._menu_hold_started_s if self._menu_hold_started_s is not None else now_s
        self._menu_hold_progress = min(1.0, (now_s - started_s) / self.menu_hold_s)
        if self._menu_hold_progress >= 1.0:
            if action == "select":
                self._select_menu_item()
            else:
                self._back_menu()
            self._menu_tilt_armed = False
            self._reset_menu_hold()

    def handle_key(self, key: int) -> bool:
        if key == 255:
            return False
        if self._screen == SnakeScreen.EXITING:
            return True
        if key in (ord("q"), ord("Q")):
            self.request_quit()
            return True
        if key in (ord("w"), ord("W")):
            self._move_menu(-1)
            return True
        if key in (ord("s"), ord("S")):
            self._move_menu(1)
            return True
        if key in (13, 10):
            self._select_menu_item()
            return True
        if key in (27, 8):
            if self._screen == SnakeScreen.PLAYING:
                self.open_pause_menu()
                return True
            self._back_menu()
            return True
        if key in (ord("m"), ord("M")):
            if self._screen == SnakeScreen.PLAYING:
                self.open_pause_menu()
            elif self._screen == SnakeScreen.MAIN_MENU and self._in_game_menu:
                self._resume_game()
            else:
                self._in_game_menu = False
                self._pause_notice = None
                self._menu_index = 0
                self._screen = SnakeScreen.MAIN_MENU
                self._game_over = False
                self._paused = False
            return True
        if key in (ord("p"), ord("P")):
            self.toggle_pause()
            return True
        if key in (ord("r"), ord("R")):
            self.reset()
            return True
        return False

    def _move_menu(self, delta: int) -> None:
        if self._screen == SnakeScreen.MAIN_MENU:
            old_index = self._menu_index
            self._menu_index = (self._menu_index + delta) % len(self._main_menu_items)
            if self._menu_index != old_index:
                self._queue_sound("menu_move")
        elif self._screen == SnakeScreen.GAME_OVER:
            old_index = self._game_over_index
            self._game_over_index = (self._game_over_index + delta) % len(self._game_over_items)
            if self._game_over_index != old_index:
                self._queue_sound("menu_move")

    def _select_menu_item(self) -> None:
        if self._screen == SnakeScreen.MAIN_MENU:
            item = self._main_menu_items[self._menu_index]
            if item == "Resume Game":
                self._queue_sound("menu_select")
                self._resume_game()
            elif item == "Start Game":
                self._queue_sound("menu_select")
                self.reset()
            elif item.startswith("Classic Walls"):
                self._queue_sound("menu_select")
                self._toggle_boundary_mode()
            elif item.startswith("Sound"):
                sound_was_enabled = self._sound_enabled
                if sound_was_enabled:
                    self._queue_sound("menu_select")
                self._toggle_sound()
                if not sound_was_enabled:
                    self._queue_sound("menu_select")
            elif item.startswith("Camera Flip"):
                self._queue_sound("menu_select")
                self._toggle_camera_flip()
            elif item == "Instructions":
                self._queue_sound("menu_select")
                self._screen = SnakeScreen.INSTRUCTIONS
            elif item == "High Scores":
                self._queue_sound("menu_select")
                self._screen = SnakeScreen.HIGH_SCORES
            elif item == "About":
                self._queue_sound("menu_select")
                self._screen = SnakeScreen.ABOUT
            elif item == "Return to Main Menu":
                self._queue_sound("menu_select")
                self._return_to_main_menu()
            else:
                self.request_quit()
        elif self._screen == SnakeScreen.GAME_OVER:
            item = self._game_over_items[self._game_over_index]
            if item == "Restart":
                self._queue_sound("menu_select")
                self.reset()
            elif item == "Main Menu":
                self._queue_sound("menu_select")
                self._return_to_main_menu()
            elif item == "High Scores":
                self._queue_sound("menu_select")
                self._in_game_menu = False
                self._pause_notice = None
                self._screen = SnakeScreen.HIGH_SCORES
                self._game_over = False
            elif item == "About":
                self._queue_sound("menu_select")
                self._in_game_menu = False
                self._pause_notice = None
                self._screen = SnakeScreen.ABOUT
                self._game_over = False
            elif item == "Quit":
                self.request_quit()

    def _back_menu(self) -> None:
        if self._screen in (SnakeScreen.INSTRUCTIONS, SnakeScreen.HIGH_SCORES, SnakeScreen.ABOUT):
            self._queue_sound("menu_back")
            self._screen = SnakeScreen.MAIN_MENU
        elif self._screen == SnakeScreen.MAIN_MENU and self._in_game_menu:
            self._queue_sound("menu_back")
            self._resume_game()
        elif self._screen == SnakeScreen.GAME_OVER:
            self._queue_sound("menu_back")
            self._return_to_main_menu()

    def _reset_menu_hold(self) -> None:
        self._menu_hold_action = None
        self._menu_hold_started_s = None
        self._menu_hold_progress = 0.0

    def _toggle_boundary_mode(self) -> None:
        self._boundary_mode = BoundaryMode.WALLS if self._boundary_mode == BoundaryMode.BOUNDARYLESS else BoundaryMode.BOUNDARYLESS

    def _toggle_sound(self) -> None:
        self._sound_enabled = not self._sound_enabled

    def _toggle_camera_flip(self) -> None:
        self.camera_flipped = not self.camera_flipped

    def _resume_game(self) -> None:
        self._screen = SnakeScreen.PLAYING
        self._paused = False
        self._pause_notice = None
        self._reset_menu_hold()

    def _return_to_main_menu(self) -> None:
        self._in_game_menu = False
        self._pause_notice = None
        self._screen = SnakeScreen.MAIN_MENU
        self._game_over = False
        self._paused = False
        self._menu_index = 0
        self._game_over_index = 0
        self._reset_menu_hold()

    def _start_exit(self) -> None:
        self._screen = SnakeScreen.EXITING
        self._game_over = False
        self._paused = False
        self._in_game_menu = False
        self._pause_notice = None
        self._exit_started_s = None
        self._quit_requested = False
        self._reset_menu_hold()
        self._queue_sound("quit_hiss")

    def _queue_sound(self, sound_name: str) -> None:
        if not self._sound_enabled:
            return
        self._sound_events.append(sound_name)

    def _menu_action_from_pose(self, *, dx: float, dy: float, roll_delta_deg: float) -> Optional[str]:
        allow_select = self._screen in (SnakeScreen.MAIN_MENU, SnakeScreen.GAME_OVER)
        allow_back = self._screen != SnakeScreen.MAIN_MENU
        if hypot(dx, dy) >= self.menu_side_threshold and abs(dy) <= abs(dx):
            if dx > 0 and allow_select:
                return "select"
            if dx < 0 and allow_back:
                return "back"

        if roll_delta_deg >= self.menu_tilt_threshold_deg and allow_select:
            return "select"
        if roll_delta_deg <= -self.menu_tilt_threshold_deg and allow_back:
            return "back"
        return None

    def _menu_action_released(self, *, dx: float, dy: float, roll_delta_deg: float) -> bool:
        return hypot(dx, dy) <= self.menu_side_release_threshold and abs(roll_delta_deg) <= self.menu_tilt_release_deg

    def set_direction(self, direction: GazeDirection) -> None:
        if not self.accepts_direction_input:
            return
        if direction == GazeDirection.CENTER:
            return
        if self._is_opposite(direction, self._direction):
            return
        self._pending_direction = direction

    def update(self, now_s: float) -> None:
        if self._screen == SnakeScreen.EXITING:
            if self._exit_started_s is None:
                self._exit_started_s = now_s
            elif (now_s - self._exit_started_s) >= self.exit_message_s:
                self._quit_requested = True
            return
        if self._screen != SnakeScreen.PLAYING or self._game_over or self._paused:
            return
        if self._last_step_time_s and (now_s - self._last_step_time_s) < self.step_s:
            return

        self._expire_bonus_food(now_s)
        self._last_step_time_s = now_s
        if self._is_grid_aligned(self._snake[0]):
            self._apply_pending_direction()

        dx, dy = self._direction_delta(self._direction)
        head_x, head_y = self._snake[0]
        raw_next_head = (head_x + dx, head_y + dy)
        if self._boundary_mode == BoundaryMode.WALLS and self._unit_hits_wall(raw_next_head):
            self._finish_game()
            return
        next_head = raw_next_head if self._boundary_mode == BoundaryMode.WALLS else self._wrap_unit(raw_next_head)
        eating = self._is_grid_aligned(next_head) and self._unit_to_grid(next_head) == self._food
        eating_bonus = self._is_grid_aligned(next_head) and self._bonus_food is not None and self._unit_to_grid(next_head) in self._bonus_food.cells
        growing = eating or self._growth_units > 0
        collision_body = set(self._snake if growing else list(self._snake)[:-1])

        if next_head in collision_body:
            self._finish_game()
            return

        self._snake.appendleft(next_head)
        if eating_bonus:
            self._score += self.bonus_food_score
            self._bonus_food = None
            self._queue_sound("bonus_food_pickup")
        elif eating:
            self._score += 1
            self._normal_foods_since_bonus += 1
            self._growth_units += self.movement_units_per_cell
            self._food = self._spawn_food()
            self._queue_sound("food_pickup")
            if self._normal_foods_since_bonus >= self.bonus_food_every:
                self._bonus_food = self._spawn_bonus_food(now_s)
                self._normal_foods_since_bonus = 0

        if self._growth_units > 0:
            self._growth_units -= 1
        else:
            self._snake.pop()

        if self._is_grid_aligned(next_head):
            self._apply_pending_direction()

    def render(
        self,
        *,
        tracking_available: bool,
        control_direction: GazeDirection,
        control_status: str,
        calibration_progress: float,
        dx: float,
        dy: float,
        activation_radius: float,
        release_radius: float,
        camera_preview: Optional[np.ndarray] = None,
        target_size: Optional[Tuple[int, int]] = None,
        now_s: Optional[float] = None,
    ) -> np.ndarray:
        width = self.grid_width * self.cell_size
        header_height = 124
        height = self.grid_height * self.cell_size + header_height
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = (52, 30, 46)

        board = canvas[header_height:, :]
        board[:, :] = (48, 36, 58)

        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if (x + y) % 2 == 0:
                    x0 = x * self.cell_size
                    y0 = y * self.cell_size
                    cv2.rectangle(board, (x0, y0), (x0 + self.cell_size, y0 + self.cell_size), (58, 44, 72), -1)

        if self._boundary_mode == BoundaryMode.WALLS:
            self._draw_boundary_walls(board)

        fx, fy = self._food
        self._draw_cell(board, (fx, fy), (40, 70, 255), radius_scale=0.45)
        if self._bonus_food is not None:
            for point in self._bonus_food.cells:
                self._draw_cell(board, point, (60, 220, 255), radius_scale=0.42)
            self._draw_bonus_timer(board, now_s)

        snake_points: List[UnitPoint] = list(self._snake)
        for idx, point in enumerate(snake_points):
            color = (80, 245, 210) if idx == 0 else (0, 150, 90)
            self._draw_unit_point(board, point, color)

        face_status = "FACE LOCKED" if tracking_available else control_status.upper()
        self._draw_pixel_text(canvas, f"SCORE {self._score}  BEST {self.high_score}", (16, 16), 15, (255, 255, 255))
        self._draw_pixel_text(canvas, f"INPUT {control_direction.value}", (16, 48), 13, (0, 200, 255))
        self._draw_pixel_text(canvas, face_status, (16, 78), 9, (180, 255, 180) if tracking_available else (70, 170, 255))
        self._draw_pixel_text(canvas, f"MODE {self._boundary_mode.value.upper()}", (16, 102), 9, (210, 210, 210))

        pad_center = (width - 178, 62)
        pad_radius = 34
        pad_value_radius = 0.18
        movement_radius = int(max(4.0, min(1.0, activation_radius / pad_value_radius) * pad_radius))
        neutral_radius = int(max(4.0, min(1.0, release_radius / pad_value_radius) * pad_radius))
        pointer_x = int(pad_center[0] + max(-1.0, min(1.0, dx / pad_value_radius)) * pad_radius)
        pointer_y = int(pad_center[1] + max(-1.0, min(1.0, dy / pad_value_radius)) * pad_radius)
        cv2.circle(canvas, pad_center, pad_radius, (90, 100, 105), 1)
        cv2.circle(canvas, pad_center, neutral_radius, (55, 70, 75), -1)
        cv2.circle(canvas, pad_center, neutral_radius, (115, 130, 135), 1)
        cv2.circle(canvas, pad_center, movement_radius, (0, 165, 220), 1)
        cv2.line(canvas, (pad_center[0] - pad_radius, pad_center[1]), (pad_center[0] + pad_radius, pad_center[1]), (70, 80, 85), 1)
        cv2.line(canvas, (pad_center[0], pad_center[1] - pad_radius), (pad_center[0], pad_center[1] + pad_radius), (70, 80, 85), 1)
        cv2.circle(canvas, (pointer_x, pointer_y), 6, (0, 210, 255), -1)

        if calibration_progress < 1.0:
            bar_x, bar_y, bar_w, bar_h = 210, 102, 130, 10
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + int(bar_w * calibration_progress), bar_y + bar_h), (0, 180, 255), -1)

        if camera_preview is not None:
            preview_h = 90
            preview_w = 120
            preview = cv2.resize(camera_preview, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
            px = width - preview_w - 12
            py = 12
            canvas[py : py + preview_h, px : px + preview_w] = preview
            cv2.rectangle(canvas, (px, py), (px + preview_w, py + preview_h), (190, 190, 190), 1)

        if self._screen == SnakeScreen.EXITING:
            self._draw_goodbye(board)
        elif self._paused:
            self._draw_center_message(board, "PAUSED", "Press P to resume")
        elif self._game_over:
            self._draw_game_over_menu(board)
        elif self._screen == SnakeScreen.MAIN_MENU:
            self._draw_menu_panel(board, "SNAKINESIS", "HEAD UP/DOWN | HOLD RIGHT TO SELECT", self._main_menu_items, self._menu_index)
        elif self._screen == SnakeScreen.INSTRUCTIONS:
            self._draw_instructions(board)
        elif self._screen == SnakeScreen.HIGH_SCORES:
            self._draw_high_scores(board)
        elif self._screen == SnakeScreen.ABOUT:
            self._draw_about(board)

        return self._letterbox(canvas, target_size)

    def _spawn_food(self) -> GridPoint:
        occupied = {self._unit_to_grid(point) for point in self._snake if self._is_grid_aligned(point)}
        if self._bonus_food is not None:
            occupied.update(self._bonus_food.cells)
        candidates = [
            (x, y)
            for y in range(self.grid_height)
            for x in range(self.grid_width)
            if self._cell_is_playable((x, y)) and (x, y) not in occupied
        ]
        if candidates:
            return random.choice(candidates)
        return (self.grid_width // 2, self.grid_height // 2)

    def _spawn_bonus_food(self, now_s: float) -> Optional[BonusFood]:
        occupied = {self._unit_to_grid(point) for point in self._snake if self._is_grid_aligned(point)}
        occupied.add(self._food)
        candidates: List[Tuple[GridPoint, ...]] = []
        for y in range(self.grid_height - 1):
            for x in range(self.grid_width - 1):
                cells = ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))
                if all(self._cell_is_playable(cell) for cell in cells) and not any(cell in occupied for cell in cells):
                    candidates.append(cells)
        if not candidates:
            return None
        return BonusFood(cells=random.choice(candidates), expires_at_s=now_s + self.bonus_food_duration_s)

    def _expire_bonus_food(self, now_s: float) -> None:
        if self._bonus_food is not None and now_s >= self._bonus_food.expires_at_s:
            self._bonus_food = None

    def _finish_game(self) -> None:
        self._game_over = True
        self._paused = False
        self._screen = SnakeScreen.GAME_OVER
        self._save_high_score()
        self._queue_sound("death")

    @property
    def _unit_width(self) -> int:
        return self.grid_width * self.movement_units_per_cell

    @property
    def _unit_height(self) -> int:
        return self.grid_height * self.movement_units_per_cell

    @property
    def _main_menu_items(self) -> Tuple[str, ...]:
        sound_state = "ON" if self._sound_enabled else "OFF"
        flip_state = "ON" if self.camera_flipped else "OFF"
        if self._in_game_menu:
            return (
                "Resume Game",
                f"Sound: {sound_state}",
                f"Camera Flip: {flip_state}",
                "Instructions",
                "High Scores",
                "About",
                "Return to Main Menu",
                "Quit",
            )
        walls_state = "ON" if self._boundary_mode == BoundaryMode.WALLS else "OFF"
        return (
            "Start Game",
            f"Classic Walls: {walls_state}",
            f"Sound: {sound_state}",
            f"Camera Flip: {flip_state}",
            "Instructions",
            "High Scores",
            "About",
            "Quit",
        )

    @property
    def _game_over_items(self) -> Tuple[str, ...]:
        return ("Restart", "Main Menu", "High Scores", "About", "Quit")

    def _apply_pending_direction(self) -> None:
        if not self._is_opposite(self._pending_direction, self._direction):
            self._direction = self._pending_direction

    def _wrap_unit(self, point: UnitPoint) -> UnitPoint:
        return (point[0] % self._unit_width, point[1] % self._unit_height)

    def _unit_in_bounds(self, point: UnitPoint) -> bool:
        return 0 <= point[0] < self._unit_width and 0 <= point[1] < self._unit_height

    def _unit_hits_wall(self, point: UnitPoint) -> bool:
        return not self._unit_in_bounds(point) or self._is_wall_cell(self._unit_to_grid(point))

    def _is_wall_cell(self, point: GridPoint) -> bool:
        x, y = point
        return x == 0 or y == 0 or x == self.grid_width - 1 or y == self.grid_height - 1

    def _cell_is_playable(self, point: GridPoint) -> bool:
        return self._boundary_mode != BoundaryMode.WALLS or not self._is_wall_cell(point)

    def _is_grid_aligned(self, point: UnitPoint) -> bool:
        return point[0] % self.movement_units_per_cell == 0 and point[1] % self.movement_units_per_cell == 0

    def _unit_to_grid(self, point: UnitPoint) -> GridPoint:
        return (
            (point[0] // self.movement_units_per_cell) % self.grid_width,
            (point[1] // self.movement_units_per_cell) % self.grid_height,
        )

    def _mode_key(self, mode: BoundaryMode) -> str:
        return "walls" if mode == BoundaryMode.WALLS else "boundaryless"

    def _load_high_scores(self) -> dict:
        defaults = {"boundaryless": 0, "walls": 0}
        if not self.high_score_path:
            return defaults
        path = Path(self.high_score_path)
        if not path.exists():
            return defaults
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return defaults
        return {key: int(loaded.get(key, 0)) for key in defaults}

    def _save_high_score(self) -> None:
        key = self._mode_key(self._boundary_mode)
        if self._score <= int(self._high_scores.get(key, 0)):
            return
        self._high_scores[key] = self._score
        if not self.high_score_path:
            return
        path = Path(self.high_score_path)
        try:
            path.write_text(json.dumps(self._high_scores, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _draw_menu_panel(self, board: np.ndarray, title: str, subtitle: str, items: Tuple[str, ...], selected_index: int) -> None:
        height, width = board.shape[:2]
        overlay = board.copy()
        cv2.rectangle(overlay, (44, 44), (width - 44, height - 22), (24, 16, 34), -1)
        board[:] = cv2.addWeighted(overlay, 0.78, board, 0.22, 0.0)
        if title == "SNAKINESIS":
            self._draw_logo(board, (width - 176, 70), 96)
        self._draw_pixel_text(board, title, (78, 88), 24, (255, 245, 235))
        self._draw_pixel_text(board, subtitle, (78, 132), 9, (210, 220, 235))
        if self._pause_notice:
            self._draw_pixel_text(board, self._pause_notice.upper(), (78, 160), 9, (70, 230, 255))

        start_y = 210
        for idx, item in enumerate(items):
            y = start_y + idx * 46
            selected = idx == selected_index
            color = (40, 210, 255) if selected else (225, 220, 230)
            if selected:
                cv2.rectangle(board, (74, y - 32), (width - 74, y + 10), (74, 50, 92), -1)
            self._draw_pixel_text(board, item.upper(), (96, y - 20), 13, color)

        warning = self._selected_menu_warning()
        if warning:
            warning_y = 184 if self._pause_notice else 164
            cv2.rectangle(board, (74, warning_y - 12), (width - 74, warning_y + 26), (72, 36, 58), -1)
            self._draw_pixel_text(board, warning, (96, warning_y), 9, (70, 170, 255))

        self._draw_hold_progress(board, width, height)

    def _draw_instructions(self, board: np.ndarray) -> None:
        height, width = board.shape[:2]
        overlay = board.copy()
        cv2.rectangle(overlay, (32, 34), (width - 32, height - 34), (18, 18, 30), -1)
        board[:] = cv2.addWeighted(overlay, 0.84, board, 0.16, 0.0)
        self._draw_pixel_text(board, "HOW TO PLAY", (70, 70), 18, (255, 245, 235))

        lines = (
            "HEAD LEFT/RIGHT/UP/DOWN TO TURN",
            "RETURN TO CENTER BEFORE NEXT MOVE",
            "MENU: HEAD UP/DOWN TO CHOOSE",
            "HOLD RIGHT TO SELECT",
            "HOLD LEFT TO GO BACK",
            "C = RECALIBRATE IF DOT DRIFTS",
            "P/ESC = PAUSE MENU",
            "R = RESTART   Q = QUIT",
            "COVER FACE BRIEFLY = AUTO PAUSE",
            "WALLS ON: PURPLE BORDER KILLS",
            "WALLS OFF: SNAKE WRAPS EDGES",
            f"BONUS FOOD: +{self.bonus_food_score}, NO GROWTH",
        )
        y = 128
        for line in lines:
            self._draw_pixel_text(board, line, (70, y), 9, (220, 228, 238))
            y += 31

        self._draw_pixel_text(board, "PRESS ESC OR HOLD LEFT TO RETURN", (70, height - 82), 9, (40, 210, 255))
        self._draw_hold_progress(board, width, height)

    def _draw_high_scores(self, board: np.ndarray) -> None:
        height, width = board.shape[:2]
        overlay = board.copy()
        cv2.rectangle(overlay, (44, 54), (width - 44, height - 54), (18, 18, 28), -1)
        board[:] = cv2.addWeighted(overlay, 0.80, board, 0.20, 0.0)
        self._draw_pixel_text(board, "HIGH SCORES", (86, 95), 20, (255, 245, 235))
        self._draw_pixel_text(board, f"BOUNDARYLESS: {self._high_scores.get('boundaryless', 0)}", (94, 195), 12, (70, 230, 255))
        self._draw_pixel_text(board, f"CLASSIC WALLS: {self._high_scores.get('walls', 0)}", (94, 252), 12, (70, 230, 255))
        self._draw_pixel_text(board, "HOLD LEFT OR PRESS ESC", (94, height - 102), 9, (220, 220, 230))
        self._draw_hold_progress(board, width, height)

    def _draw_about(self, board: np.ndarray) -> None:
        height, width = board.shape[:2]
        overlay = board.copy()
        cv2.rectangle(overlay, (44, 54), (width - 44, height - 54), (18, 18, 28), -1)
        board[:] = cv2.addWeighted(overlay, 0.80, board, 0.20, 0.0)
        self._draw_pixel_text(board, "ABOUT", (86, 88), 22, (255, 245, 235))
        self._draw_pixel_text(board, f"SNAKINESIS V{APP_VERSION}", (94, 150), 12, (70, 230, 255))
        self._draw_pixel_text(board, "A HANDS-FREE SNAKE GAME", (94, 196), 9, (220, 228, 238))
        self._draw_pixel_text(board, "AUTHORS", (94, 258), 12, (255, 245, 235))

        y = 312
        for name, link in AUTHORS:
            self._draw_pixel_text(board, name.upper(), (94, y), 10, (220, 228, 238))
            self._draw_pixel_text(board, link, (94, y + 28), 8, (70, 230, 255))
            y += 72

        self._draw_pixel_text(board, "HOLD LEFT OR PRESS ESC", (94, height - 102), 9, (220, 220, 230))
        self._draw_hold_progress(board, width, height)

    def _draw_game_over_menu(self, board: np.ndarray) -> None:
        subtitle = f"Score {self._score} | Best {self.high_score} | {self._boundary_mode.value}"
        self._draw_menu_panel(board, "GAME OVER", subtitle, self._game_over_items, self._game_over_index)

    def _draw_goodbye(self, board: np.ndarray) -> None:
        height, width = board.shape[:2]
        overlay = board.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (18, 12, 28), -1)
        board[:] = cv2.addWeighted(overlay, 0.88, board, 0.12, 0.0)
        self._draw_centered_pixel_text(board, "GOODBYE", height // 2 - 52, 24, (255, 245, 235))
        self._draw_centered_pixel_text(board, "HISS YOU LATER", height // 2 + 8, 13, (40, 210, 255))

    def _selected_menu_warning(self) -> Optional[str]:
        if self._screen != SnakeScreen.MAIN_MENU or not self._in_game_menu:
            return None
        if self._main_menu_items[self._menu_index] == "Return to Main Menu":
            return "ALL PROGRESS WILL BE LOST"
        return None

    def _draw_hold_progress(self, board: np.ndarray, width: int, height: int) -> None:
        if self._menu_hold_action is None or self._menu_hold_progress <= 0.0:
            return
        bar_w = min(300, width - 170)
        bar_h = 10
        x0 = (width - bar_w) // 2
        y0 = 12
        cv2.rectangle(board, (x0 - 6, y0 - 6), (x0 + bar_w + 6, y0 + bar_h + 6), (24, 16, 34), -1)
        cv2.rectangle(board, (x0, y0), (x0 + bar_w, y0 + bar_h), (92, 84, 112), 1)
        cv2.rectangle(board, (x0, y0), (x0 + int(bar_w * self._menu_hold_progress), y0 + bar_h), (30, 205, 255), -1)

    def _draw_boundary_walls(self, board: np.ndarray) -> None:
        wall_color = (118, 74, 158)
        wall_highlight = (165, 110, 210)
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if not self._is_wall_cell((x, y)):
                    continue
                x0 = x * self.cell_size
                y0 = y * self.cell_size
                cv2.rectangle(board, (x0, y0), (x0 + self.cell_size, y0 + self.cell_size), wall_color, -1)
                cv2.rectangle(board, (x0 + 3, y0 + 3), (x0 + self.cell_size - 3, y0 + self.cell_size - 3), wall_highlight, 1)

    def _draw_bonus_timer(self, board: np.ndarray, now_s: Optional[float]) -> None:
        if self._bonus_food is None:
            return
        current_s = self._bonus_food.expires_at_s - self.bonus_food_duration_s if now_s is None else now_s
        remaining_s = max(0.0, self._bonus_food.expires_at_s - current_s)
        progress = max(0.0, min(1.0, remaining_s / self.bonus_food_duration_s))
        width = board.shape[1]
        bar_w = min(220, width - 96)
        bar_h = 12
        x0 = (width - bar_w) // 2
        y0 = 12
        cv2.rectangle(board, (x0 - 6, y0 - 6), (x0 + bar_w + 6, y0 + 34), (34, 22, 44), -1)
        cv2.rectangle(board, (x0, y0), (x0 + bar_w, y0 + bar_h), (105, 90, 118), 1)
        cv2.rectangle(board, (x0, y0), (x0 + int(bar_w * progress), y0 + bar_h), (40, 210, 255), -1)
        self._draw_pixel_text(board, f"BONUS {remaining_s:.1f}S", (x0, y0 + 20), 9, (235, 240, 245))

    def _draw_unit_point(self, board: np.ndarray, point: UnitPoint, color: Tuple[int, int, int]) -> None:
        center = (
            int((point[0] / self.movement_units_per_cell) * self.cell_size + self.cell_size / 2),
            int((point[1] / self.movement_units_per_cell) * self.cell_size + self.cell_size / 2),
        )
        radius = max(4, int(self.cell_size * 0.42))
        cv2.circle(board, center, radius, color, -1)

    def _draw_cell(self, board: np.ndarray, point: GridPoint, color: Tuple[int, int, int], radius_scale: float = 0.48) -> None:
        x, y = point
        x0 = x * self.cell_size
        y0 = y * self.cell_size
        center = (x0 + self.cell_size // 2, y0 + self.cell_size // 2)
        radius = max(4, int(self.cell_size * radius_scale))
        cv2.circle(board, center, radius, color, -1)

    @staticmethod
    def _letterbox(image: np.ndarray, target_size: Optional[Tuple[int, int]]) -> np.ndarray:
        if target_size is None:
            return image

        target_width, target_height = target_size
        if target_width <= 0 or target_height <= 0:
            return image

        source_height, source_width = image.shape[:2]
        scale = min(target_width / source_width, target_height / source_height)
        scaled_width = max(1, int(round(source_width * scale)))
        scaled_height = max(1, int(round(source_height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        scaled = cv2.resize(image, (scaled_width, scaled_height), interpolation=interpolation)

        output = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x0 = (target_width - scaled_width) // 2
        y0 = (target_height - scaled_height) // 2
        output[y0 : y0 + scaled_height, x0 : x0 + scaled_width] = scaled
        return output

    def _draw_center_message(self, board: np.ndarray, title: str, subtitle: str) -> None:
        height, width = board.shape[:2]
        overlay = board.copy()
        cv2.rectangle(overlay, (40, height // 2 - 60), (width - 40, height // 2 + 40), (10, 10, 10), -1)
        board[:] = cv2.addWeighted(overlay, 0.65, board, 0.35, 0.0)
        self._draw_pixel_text(board, title, (width // 2 - 116, height // 2 - 28), 18, (255, 255, 255))
        self._draw_pixel_text(board, subtitle, (width // 2 - 152, height // 2 + 18), 9, (220, 220, 220))

    @staticmethod
    def _draw_logo(image: np.ndarray, position: Tuple[int, int], size: int) -> None:
        logo = _logo_image()
        if logo is None:
            return
        resized = logo.resize((size, size), Image.Resampling.LANCZOS)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("RGBA")
        pil_image.alpha_composite(resized, dest=position)
        image[:] = cv2.cvtColor(np.asarray(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _draw_pixel_text(image: np.ndarray, text: str, position: Tuple[int, int], size: int, color: Tuple[int, int, int]) -> None:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=_pixel_font(size), fill=rgb_color)
        image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _draw_centered_pixel_text(image: np.ndarray, text: str, y: int, size: int, color: Tuple[int, int, int]) -> None:
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        font = _pixel_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = (image.shape[1] - text_width) // 2
        rgb_color = (color[2], color[1], color[0])
        draw.text((x, y), text, font=font, fill=rgb_color)
        image[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _direction_delta(direction: GazeDirection) -> GridPoint:
        if direction == GazeDirection.LEFT:
            return (-1, 0)
        if direction == GazeDirection.RIGHT:
            return (1, 0)
        if direction == GazeDirection.UP:
            return (0, -1)
        if direction == GazeDirection.DOWN:
            return (0, 1)
        return (0, 0)

    @staticmethod
    def _is_opposite(a: GazeDirection, b: GazeDirection) -> bool:
        opposites = {
            (GazeDirection.LEFT, GazeDirection.RIGHT),
            (GazeDirection.RIGHT, GazeDirection.LEFT),
            (GazeDirection.UP, GazeDirection.DOWN),
            (GazeDirection.DOWN, GazeDirection.UP),
        }
        return (a, b) in opposites
