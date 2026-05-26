# Snakinesis

**Snakinesis** is a hands-free, webcam-controlled Snake game. Instead of keyboard-only movement, the player steers by making deliberate head movements detected through computer vision.

The project is built with Python, OpenCV, MediaPipe FaceMesh, NumPy, and Pillow. It is designed as a polished thesis/demo game with a retro pixel-art interface, menu system, high scores, bonus food, and accessible fallback keyboard controls.

## Highlights

- Hands-free Snake controlled by head movement through a normal webcam
- Calibrated face-center tracking instead of unreliable iris/eye-gaze control
- One-shot movement gestures: move once, return to neutral, then move again
- Boundaryless wrap mode and optional Classic Walls mode
- Main menu, pause menu, instructions pane, about pane, high scores, and game-over menu
- Timed 2x2 bonus food that gives extra score without growing the snake
- Auto-pause with separate messages for covered face vs. face out of frame
- Sound effects for menu browsing, menu select/back, pause, normal food, bonus food, death, and hissing quit
- Graceful goodbye screen: `GOODBYE` / `HISS YOU LATER`
- Retro pixel-art logo, icon, color palette, and Press Start 2P font
- Resizable OpenCV window with proportional letterboxing
- Automated unit tests for control logic, game rules, menus, and rendering helpers

## Gameplay

The goal is classic Snake: collect food, grow longer, and avoid colliding with yourself. The game updates on a grid, but the snake moves in half-cell increments for smoother motion.

### Modes

- **Boundaryless**: the snake wraps around the edges of the board.
- **Classic Walls**: the outermost grid layer becomes a wall; hitting it ends the game.

Classic Walls can be toggled from the main menu. The pause menu intentionally does not allow mode changes because changing wall rules mid-run would make the current score/state unfair.

### Food

- **Normal food** gives `+1` score and grows the snake.
- **Bonus food** appears periodically as a larger 2x2 target.
- Bonus food gives a larger score reward and does **not** grow the snake.
- Bonus food is temporary and displays its own timer bar.

### High Scores

High scores are stored separately for Boundaryless and Classic Walls mode in `snake_high_scores.json`. That file is ignored by Git because it is local player data.

### Pause And Exit

- Press `P` or `Esc` during gameplay to open the pause menu.
- Losing face tracking briefly during gameplay opens the pause menu automatically, with separate notices for a covered face and a face that moved out of frame.
- The pause menu includes `Resume Game`, `Sound`, `Camera Flip`, `Instructions`, `High Scores`, `About`, `Return to Main Menu`, and `Quit`.
- `Return to Main Menu` warns that all current progress will be lost.
- Quitting shows a short goodbye screen before the app closes.

### About

The About pane shows the current version and project authors:

- Muhammad Umar Nadeem — `https://github.com/umrndem`
- Shifa Zeeshan — `https://github.com/AshwaZeeshan`

### Sound Effects

- Menu movement plays a short retro navigation blip.
- Menu selection plays a brighter confirmation cue.
- Backing out of submenus or the pause menu plays a descending back cue.
- Pausing the game plays a short pause cue.
- Food pickup plays a bright reward chime.
- Bonus food pickup plays a bigger jackpot chime.
- Self-collision or wall collision plays a death cue.
- Quitting plays a short snake hiss.
- Sound can be toggled ON/OFF directly from both the main menu and pause menu.

Sound playback uses the cross-platform `sounddevice` package and bundled WAV files. The quit hiss is a public-domain rattlesnake WAV listed by Parallax Learn with original source from U.S. Fish & Wildlife.

## Controls

### Head Controls

| Action | Gesture |
| --- | --- |
| Turn left | Move head left past the center radius |
| Turn right | Move head right past the center radius |
| Turn up | Move head up past the center radius |
| Turn down | Move head down past the center radius |
| Re-arm next movement | Return head to the neutral center zone |
| Menu select | Hold head/right tilt to the right |
| Menu back | Hold head/left tilt to the left |
| Auto-pause | Cover face or move fully out of frame briefly |

The small tracking pad in the UI shows the live head position. A movement is registered only when the tracking dot crosses the configured activation radius. After a movement fires, the dot must return to the neutral zone before another movement can trigger.

`Camera Flip` can be toggled from the menus if a webcam already provides a mirrored feed or if left/right movement feels reversed. Toggling it resets calibration automatically on the next frame.

### Keyboard Fallbacks

| Key | Action |
| --- | --- |
| `W` / `S` | Move menu selection up/down |
| `Enter` | Select menu item |
| `Esc` | Open pause menu during gameplay, or go back in menus |
| `P` | Pause/resume through the pause menu |
| `R` | Restart the current game |
| `C` | Recalibrate head center |
| `Q` | Show goodbye screen and quit |

## Computer Vision Design

Snakinesis uses MediaPipe FaceMesh landmarks to estimate the face center and head roll.

The current tracking design intentionally avoids eye-gaze movement because iris/eye tracking was too jittery for a fast Snake game. Deliberate head movement is more stable, more forgiving, and easier to calibrate with a webcam.

Pipeline:

1. `snakinesis/app.py` captures webcam frames with OpenCV.
2. MediaPipe FaceMesh detects face landmarks.
3. `snakinesis/tracker.py` computes smoothed face-center ratios and head roll.
4. `snakinesis/snake_game/head_control.py` compares live head position against the calibrated baseline.
5. `snakinesis/snake_game/game.py` applies one-shot movement commands, menu commands, and game rules.

## Project Structure

```text
.
├── run.py
├── requirements.txt
├── setup.ps1
├── assets/
│   ├── snakinesis_logo.png
│   ├── snakinesis.ico
│   ├── sounds/
│   │   ├── bonus_food_pickup.wav
│   │   ├── death.wav
│   │   ├── food_pickup.wav
│   │   ├── menu_back.wav
│   │   ├── menu_move.wav
│   │   ├── menu_select.wav
│   │   ├── NOTICE.txt
│   │   ├── pause.wav
│   │   └── quit_hiss.wav
│   └── fonts/
│       ├── PressStart2P-Regular.ttf
│       └── OFL-PressStart2P.txt
├── snakinesis/
│   ├── app.py
│   ├── config.py
│   ├── gesture.py
│   ├── landmarks.py
│   ├── math_utils.py
│   ├── sound.py
│   ├── tracker.py
│   └── snake_game/
│       ├── game.py
│       ├── head_control.py
│       └── __init__.py
└── tests/
    └── test_snake_game.py
```

### Main Files

- `run.py`: application entry point.
- `snakinesis/app.py`: webcam loop, FaceMesh integration, OpenCV window handling, icon setup, and app lifecycle.
- `snakinesis/config.py`: central configuration for camera, game speed, grid size, tracking sensitivity, menus, and timing.
- `snakinesis/tracker.py`: converts FaceMesh landmarks into smoothed face-center and roll measurements.
- `snakinesis/sound.py`: lightweight cross-platform sound-effect playback for bundled WAV files.
- `snakinesis/snake_game/head_control.py`: calibration, neutral zone handling, movement activation, one-shot gesture re-arming, and menu tilt signals.
- `snakinesis/snake_game/game.py`: Snake state machine, collision rules, food spawning, menus, high scores, rendering, pause logic, and graceful exit.
- `snakinesis/landmarks.py`: MediaPipe landmark indices used by the tracker.
- `snakinesis/math_utils.py`: small reusable math helpers.
- `tests/test_snake_game.py`: regression tests for tracking, controls, gameplay rules, menus, high scores, and rendering helpers.

## Requirements

- Python 3.12
- Webcam
- Windows/macOS/Linux

Python dependencies:

- `mediapipe==0.10.21`
- `opencv-python==4.11.0.86`
- `numpy==1.26.4`
- `pillow>=10.0.0`
- `sounddevice==0.5.5`

PyAutoGUI is not required. Earlier desktop-control experiments used it conceptually, but the current project is a standalone Snake game and does not depend on desktop automation.

Sound effects are bundled WAV files and are played through `sounddevice` for cross-platform source compatibility.

## Install

### Windows Quick Setup

From PowerShell in the project folder:

```powershell
.\setup.ps1
.\.venv\Scripts\Activate.ps1
python run.py
```

`setup.ps1` recreates `.venv`, upgrades `pip`, and installs `requirements.txt`.

### Manual Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
```

On macOS/Linux, use the platform equivalent:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
```

## First Run

1. Sit centered in front of the webcam.
2. Launch with `python run.py`.
3. Keep your face relaxed and centered while calibration completes.
4. Watch the tracking pad in the header.
5. Move your head deliberately past the center radius to navigate and play.
6. Press `C` any time the center feels wrong.

Good lighting and a stable camera angle make tracking much more reliable.

## Configuration

Most gameplay and tracking behavior lives in `snakinesis/config.py`.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `snake_grid_width` / `snake_grid_height` | `16` / `16` | Board dimensions |
| `snake_cell_size` | `36` | Base render size per grid cell |
| `snake_step_s` | `0.22` | Snake movement interval |
| `snake_movement_units_per_cell` | `2` | Two movement units equal one visible grid cell |
| `snake_control_threshold` | `0.118` | Distance from calibrated center required to trigger movement |
| `snake_release_threshold` | `0.118` | Neutral radius required to re-arm the next movement |
| `snake_axis_bias` | `1.20` | Preference for clean horizontal/vertical gestures |
| `snake_neutral_rearm_s` | `0.10` | Time inside neutral before next command can fire |
| `snake_head_smoothing_alpha` | `0.60` | Head-position smoothing strength |
| `snake_processing_scale` | `0.55` | Lower processing resolution for faster FaceMesh |
| `snake_face_loss_pause_s` | `0.45` | Face-loss duration before auto-pause |
| `snake_bonus_food_every` | `4` | Normal foods eaten before bonus can spawn |
| `snake_bonus_food_score` | `5` | Bonus food score reward |
| `snake_bonus_food_duration_s` | `6.0` | Bonus food lifetime |
| `snake_exit_message_s` | `2.4` | Goodbye screen duration |
| `flip_selfie` | `True` | Starts with mirrored selfie camera mode enabled |

If controls feel too sensitive, increase `snake_control_threshold` and `snake_release_threshold`. If controls feel sluggish, lower them slightly.

## Testing

Run the test suite:

```powershell
python -B -m unittest discover -s tests -v
```

The tests cover:

- Face-center tracking direction changes
- Calibration behavior
- Neutral-zone re-arming
- One-shot movement gestures
- Smooth half-grid movement
- Wall and wrap collision rules
- Bonus food scoring without growth
- Menu selection/back behavior
- Pause-menu behavior
- Sound-event queueing
- Sound ON/OFF menu toggling
- High-score persistence
- Goodbye/quit flow
- Letterboxed rendering

## Troubleshooting

### Webcam Does Not Open

- Make sure no other app is using the webcam.
- Check camera permissions in the operating system.
- Try changing `camera_index` in `snakinesis/config.py` from `0` to `1`.
- If packaged as an exe, Snakinesis shows a camera error prompt instead of failing silently.

### Camera Feed Is Mirrored Incorrectly

- Toggle `Camera Flip` in the main or pause menu.
- Press `C` after changing the camera orientation if the center dot feels offset.

### Tracking Feels Off

- Press `C` to recalibrate.
- Sit centered and avoid leaning during calibration.
- Improve lighting.
- Keep the camera stable.
- Increase `snake_head_motion_scale` if small head movement does not move the dot enough.
- Increase thresholds if accidental movements happen too often.

### Game Feels Too Fast Or Slow

- Increase `snake_step_s` to slow the snake.
- Decrease `snake_step_s` to speed it up.

### Menu Selection Is Too Hard

- Lower `snake_menu_side_threshold` or `snake_menu_tilt_threshold_deg`.
- Increase `snake_menu_hold_s` if accidental menu selections happen.
- Decrease `snake_menu_hold_s` if selection feels too slow.

## Assets And Branding

- `assets/snakinesis_logo.png`: pixel-art snake logo used in the main menu.
- `assets/snakinesis.ico`: Windows title-bar/taskbar icon.
- `assets/sounds/`: retro WAV sound effects for menu movement, select/back, pause, food pickup, death, and quit hiss.
- `assets/sounds/NOTICE.txt`: sound source and public-domain attribution notes.
- `assets/fonts/PressStart2P-Regular.ttf`: bundled retro pixel font.
- `assets/fonts/OFL-PressStart2P.txt`: font license.

## Git-Ignored Local Files

The repository ignores local/generated files including:

- `.venv/`
- `__pycache__/`
- `dist/`
- `build/`
- `snake_high_scores.json`
- `PROJECT_CHANGES.TXT`
- `PROJECT_SPEC.txt`

`PROJECT_CHANGES.TXT` and `PROJECT_SPEC.txt` are local thesis/project-history documents, not runtime requirements.

## Current Status

Snakinesis is suitable as a polished thesis/demo project. For a broader public release, the next useful steps would be packaging a Windows executable, testing on multiple webcams and lighting setups, and adding an in-game sensitivity settings screen.
