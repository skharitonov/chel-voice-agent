"""Фоновый шум офиса под голос агента (фаза 2, console).

Штатный `BackgroundAudioPlayer` LiveKit публикует отдельную дорожку в комнату и
ЯВНО не работает в console-режиме (is_fake_job → варнинг). Поэтому здесь мы
подмешиваем зацикленный фоновый клип прямо во фреймы TTS агента: шум звучит за его
голосом (и попадает в запись --record) — реализм «звоню из шумного офиса».

Шум слышен, пока агент говорит (tts_node активен). Непрерывный фон в тишине
требует реальной комнаты LiveKit (dev-режим), что вне scope console-прототипа.

Громкость и клип — через окружение (см. voice_core / .env.example).
"""

from __future__ import annotations

import numpy as np
from livekit import rtc
from livekit.agents.utils.audio import audio_frames_from_file
from livekit.agents.voice.background_audio import BuiltinAudioClip

# Имя из окружения -> встроенный клип LiveKit.
CLIPS = {
    "office": BuiltinAudioClip.OFFICE_AMBIENCE,     # гул офиса
    "crowded": BuiltinAudioClip.CROWDED_ROOM,       # голоса коллег/менеджеров
    "typing": BuiltinAudioClip.KEYBOARD_TYPING,     # клавиатуры
}


class _Loop:
    """Один зацикленный клип: буфер на частоте TTS + позиция."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._buf: np.ndarray | None = None
        self._rate: int | None = None
        self._pos = 0

    async def ensure_loaded(self, sample_rate: int, num_channels: int) -> None:
        if self._buf is not None and self._rate == sample_rate:
            return
        self._rate = sample_rate
        self._pos = 0
        parts: list[np.ndarray] = []
        async for f in audio_frames_from_file(
                self._path, sample_rate=sample_rate, num_channels=num_channels):
            parts.append(np.frombuffer(f.data, dtype=np.int16).astype(np.float32))
        self._buf = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)

    def take(self, n: int) -> np.ndarray:
        buf = self._buf
        assert buf is not None
        out = np.empty(n, dtype=np.float32)
        L = len(buf)
        filled, pos = 0, self._pos
        while filled < n:
            take = min(n - filled, L - pos)
            out[filled:filled + take] = buf[pos:pos + take]
            pos = (pos + take) % L
            filled += take
        self._pos = pos
        return out


class AmbienceMixer:
    """Подмешивает один или несколько зацикленных клипов во фреймы TTS на лету."""

    def __init__(self, clips: list[str], volume: float) -> None:
        self._loops = [_Loop(CLIPS[c].path()) for c in clips]
        self._volume = max(0.0, volume)

    async def preload(self, sample_rate: int = 24000, num_channels: int = 1) -> None:
        """Загрузить клипы заранее (до звонка), чтобы первый ход не подвисал на чтении."""
        for loop in self._loops:
            await loop.ensure_loaded(sample_rate, num_channels)

    async def mix(self, frame: rtc.AudioFrame) -> rtc.AudioFrame:
        if self._volume <= 0 or not self._loops:
            return frame
        data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
        n = len(data)
        for loop in self._loops:
            await loop.ensure_loaded(frame.sample_rate, frame.num_channels)
            data += loop.take(n) * self._volume
        np.clip(data, -32768, 32767, out=data)
        return rtc.AudioFrame(
            data=data.astype(np.int16).tobytes(),
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            samples_per_channel=frame.samples_per_channel,
        )


def make_ambience(spec: str, volume: float) -> AmbienceMixer | None:
    """Фабрика. spec — имя клипа или несколько через '+'/',' (напр. 'crowded+typing').
    None, если выключено / нет валидных клипов."""
    spec = (spec or "").strip().lower()
    if spec in ("", "off", "none", "0"):
        return None
    names = [s.strip() for s in spec.replace(",", "+").split("+") if s.strip()]
    clips = [c for c in names if c in CLIPS]
    bad = [c for c in names if c not in CLIPS]
    if bad:
        print(f"⚠ CHEL_AMBIENCE: неизвестные клипы {bad} (есть: {', '.join(CLIPS)}).")
    if not clips:
        return None
    return AmbienceMixer(clips, volume)
