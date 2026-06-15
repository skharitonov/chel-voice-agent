"""Пре-флайт проверка ключей фазы 2 (Deepgram STT + Inworld TTS).

Проверяет, что ключи в .env рабочие, ДО запуска полноценного голосового звонка:
- Deepgram: лёгкая авторизация по REST (GET /v1/projects);
- Inworld: реальный синтез короткой русской фразы через плагин LiveKit
  (заодно валидирует INWORLD_VOICE / INWORLD_TTS_MODEL / INWORLD_LANG).

Запуск:  .venv/Scripts/python.exe scripts/test_voice.py
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv  # noqa: E402

Result = tuple[str, bool, str]


async def check_deepgram() -> Result:
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key or "..." in key:
        return ("Deepgram STT", False, "DEEPGRAM_API_KEY не задан в .env")
    import aiohttp

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.get("https://api.deepgram.com/v1/projects",
                             headers={"Authorization": f"Token {key}"}) as r:
                if r.status == 200:
                    return ("Deepgram STT", True, "ключ валиден (GET /v1/projects → 200)")
                body = (await r.text())[:140]
                return ("Deepgram STT", False, f"HTTP {r.status}: {body}")
    except Exception as e:  # noqa: BLE001
        return ("Deepgram STT", False, f"{type(e).__name__}: {e}")


async def check_inworld() -> Result:
    key = os.environ.get("INWORLD_API_KEY")
    if not key or "..." in key:
        return ("Inworld TTS", False, "INWORLD_API_KEY не задан в .env")
    # Мягкая проверка формата: ключ Inworld должен быть Base64.
    note = ""
    try:
        base64.b64decode(key, validate=True)
    except (binascii.Error, ValueError):
        note = " (внимание: ключ не похож на Base64 — Inworld ждёт Base64-ключ)"

    try:
        from livekit.plugins import inworld
    except Exception as e:  # noqa: BLE001
        return ("Inworld TTS", False, f"плагин не импортируется: {e}")

    kwargs: dict = {"language": os.environ.get("INWORLD_LANG", "ru-RU")}
    if os.environ.get("INWORLD_VOICE"):
        kwargs["voice"] = os.environ["INWORLD_VOICE"]
    if os.environ.get("INWORLD_TTS_MODEL"):
        kwargs["model"] = os.environ["INWORLD_TTS_MODEL"]
    for env_name, kw in (("INWORLD_SPEAKING_RATE", "speaking_rate"),
                         ("INWORLD_TEMPERATURE", "temperature")):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            try:
                kwargs[kw] = float(raw.replace(",", "."))
            except ValueError:
                note += f" (внимание: {env_name}={raw!r} не число — игнор)"

    # Вне LiveKit-воркера у плагина нет job-контекста с HTTP-сессией, поэтому
    # передаём свою aiohttp.ClientSession (иначе RuntimeError маскируется под
    # "Connection error"). В реальном voice_core сессию даёт сам воркер.
    import aiohttp
    http = aiohttp.ClientSession()
    kwargs["http_session"] = http
    tts = inworld.TTS(**kwargs)
    stream = None
    try:
        nbytes = 0
        stream = tts.synthesize("Проверка связи. Раз, два, три.")
        async for ev in stream:
            frame = getattr(ev, "frame", None)
            if frame is not None:
                nbytes += len(frame.data)
            if nbytes > 0:
                break
        voice = kwargs.get("voice", "по умолчанию")
        if nbytes > 0:
            return ("Inworld TTS", True,
                    f"синтез ок ({nbytes} байт аудио, голос: {voice}, {kwargs['language']})" + note)
        return ("Inworld TTS", False, "синтез не вернул аудио" + note)
    except Exception as e:  # noqa: BLE001
        return ("Inworld TTS", False, f"{type(e).__name__}: {e}{note}")
    finally:
        if stream is not None:
            try:
                await stream.aclose()
            except Exception:  # noqa: BLE001
                pass
        try:
            await tts.aclose()
        except Exception:  # noqa: BLE001
            pass
        try:
            await http.close()
        except Exception:  # noqa: BLE001
            pass


async def main() -> int:
    load_dotenv(REPO / ".env")
    print("Пре-флайт проверка голосовых ключей (фаза 2)…\n")
    results = await asyncio.gather(check_deepgram(), check_inworld())
    ok = True
    for name, passed, msg in results:
        print(f"{'✓' if passed else '✗'} {name}: {msg}")
        ok = ok and passed
    print()
    if ok:
        print("Готово к запуску: python -m voice_agent.voice_core console")
        return 0
    print("Исправьте ключи/параметры в .env и повторите.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
