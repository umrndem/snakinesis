from __future__ import annotations

import sys
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None  # type: ignore[assignment]

SoundData = Tuple[np.ndarray, int]


def _resource_path(*parts: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return base.joinpath(*parts)


class SoundPlayer:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled and sd is not None
        self._sounds = {
            "menu_move": _resource_path("assets", "sounds", "menu_move.wav"),
            "menu_select": _resource_path("assets", "sounds", "menu_select.wav"),
            "menu_back": _resource_path("assets", "sounds", "menu_back.wav"),
            "pause": _resource_path("assets", "sounds", "pause.wav"),
            "food_pickup": _resource_path("assets", "sounds", "food_pickup.wav"),
            "death": _resource_path("assets", "sounds", "death.wav"),
            "quit_hiss": _resource_path("assets", "sounds", "quit_hiss.wav"),
        }
        self._cache: Dict[str, SoundData] = {}

    def play(self, sound_name: str) -> None:
        if not self.enabled or sd is None:
            return
        sound_data = self._load(sound_name)
        if sound_data is None:
            return
        samples, sample_rate = sound_data
        try:
            sd.play(samples, samplerate=sample_rate, blocking=False)
        except Exception:
            pass

    def _load(self, sound_name: str) -> Optional[SoundData]:
        if sound_name in self._cache:
            return self._cache[sound_name]

        sound_path = self._sounds.get(sound_name)
        if sound_path is None or not sound_path.exists():
            return None

        try:
            with wave.open(str(sound_path), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
        except (OSError, wave.Error):
            return None

        samples = self._decode_pcm(frames, sample_width)
        if samples is None:
            return None
        if channels > 1:
            samples = samples.reshape(-1, channels)

        sound_data = (samples, sample_rate)
        self._cache[sound_name] = sound_data
        return sound_data

    @staticmethod
    def _decode_pcm(frames: bytes, sample_width: int) -> Optional[np.ndarray]:
        if sample_width == 1:
            return ((np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0).copy()
        if sample_width == 2:
            return (np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0).copy()
        if sample_width == 4:
            return (np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0).copy()
        return None
