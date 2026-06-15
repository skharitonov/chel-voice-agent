"""Голосовое ядро (фаза 2) — LiveKit Agents, локальный console-режим.

Конвейер STT → LLM → TTS:
- STT: Deepgram Nova-3 (multilingual, русский)
- LLM: наш конвейер через ChelLLM (director/УКС/compliance из фазы 1)
- TTS: Inworld TTS (русский), официальный плагин LiveKit
- VAD/перебивание: Silero

Запуск локально (вы говорите за секретаря/ЛПР, агент отвечает голосом):

    python -m voice_agent.voice_core console
    python -m voice_agent.voice_core console --record   # + запись аудио сессии

Микрофон → STT → наш конвейер → TTS → динамики. По завершении (Ctrl+C) —
постобработка тем же extractor, что и в текстовой фазе: JSON + транскрипт в results/.
С флагом --record обе стороны (вопросы и ответы) пишутся в один файл
console-recordings/session-<ts>/audio.ogg — для последующего анализа.

Параметры звонка берутся из окружения (см. .env):
    CHEL_MODE=A|B, CHEL_PHONE, CHEL_COMPANY, CHEL_ROLE (для B),
    DEEPGRAM_MODEL, DEEPGRAM_LANG, INWORLD_VOICE, INWORLD_TTS_MODEL, INWORLD_LANG.
Ключи: DEEPGRAM_API_KEY, INWORLD_API_KEY (base64).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Windows-консоль по умолчанию cp1251 — принудительно UTF-8 (эмодзи rich, Кириллица).
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli
from livekit.agents.stt import STTError
from livekit.plugins import deepgram, inworld, silero

from . import ambience as ambience_mod
from . import compliance, extractor, objections, stress as stress_mod
from .lk_llm import ChelLLM
from .providers import make_llm
from .state import ConversationState

REPO = Path(__file__).resolve().parent.parent


def _steer_for(state: ConversationState) -> str:
    """Контекстная Inworld-steering инструкция (АНГЛ.) по стадии/тону диалога.
    Тон собеседника важнее стадии. Работает на inworld-tts-2."""
    last = next((t.text for t in reversed(state.transcript)
                 if t.role == "interlocutor"), "")
    if state.refused:
        return "[sound apologetic, polite and calm]"
    if objections.detect_irritation(last):
        return "[stay calm, polite and slightly apologetic]"
    stage = state.stage or ""
    if stage == "gatekeeper_intro":
        return "[speak warmly and confidently, friendly greeting]"
    if stage == "wrap":
        return "[sound warm and appreciative]"
    if stage.startswith("lpr"):
        return "[sound professional, clear and genuinely interested]"
    if stage.startswith("gatekeeper"):
        return "[stay calm, polite and persistent]"
    return "[speak naturally and friendly]"


class ChelAgent(Agent):
    """Реплики формирует ChelLLM; инструкции тут номинальные (не используются LLM)."""

    def __init__(self, state: ConversationState) -> None:
        super().__init__(
            instructions=(
                "Голосовой агент Челябинского завода электрооборудования. Реплики "
                "формирует кастомный конвейер (director/УКС/compliance) — см. "
                "prompts/qualify_system.md и docs/specs/."
            ),
        )
        self._state = state
        # Фоновый шум офиса под голос агента (см. ambience.py). По умолчанию office.
        self._ambience = ambience_mod.make_ambience(
            os.environ.get("CHEL_AMBIENCE", "crowded"),
            _env_float("CHEL_AMBIENCE_VOLUME") or 0.2,
        )
        # Простановка ударений (U+0301) для TTS — опционально (зависит от Inworld).
        self._stress = os.environ.get("CHEL_STRESS", "off").strip().lower() in ("on", "1", "true")
        # Эмоции/манера через Inworld steering: инструкция (АНГЛ.) в начало реплики.
        # Надёжно только на inworld-tts-2; на 1.5 теги могут читаться вслух.
        self._steer = os.environ.get("CHEL_TTS_STEER", "").strip()

    def tts_node(self, text, model_settings):
        """Стандартный TTS + (опц.) ударения и эмоции (steering) в тексте + фон офиса."""
        if self._stress or self._steer:
            text = self._transform_text(text)
        base = Agent.default.tts_node(self, text, model_settings)
        if self._ambience is None:
            return base
        return self._mix_ambience(base)

    def _steer_tag(self) -> str:
        """Steering-инструкция: пусто | литерал из env | контекстная ('auto')."""
        if not self._steer:
            return ""
        if self._steer.lower() == "auto":
            return _steer_for(self._state)
        return self._steer

    async def _transform_text(self, text):
        """Ударения (везде) + steering-инструкция (в начало реплики, англ.)."""
        tag = self._steer_tag()  # один тег на всю реплику
        first = True
        async for chunk in text:
            if self._stress:
                chunk = stress_mod.apply(chunk)
            if tag and first and chunk.strip():
                chunk = f"{tag} {chunk}"  # [instruction] в начале фразы
                first = False
            yield chunk

    async def _mix_ambience(self, frames):
        async for frame in frames:
            try:
                yield await self._ambience.mix(frame)
            except Exception:  # noqa: BLE001 — фон не критичен, отдаём чистый фрейм
                yield frame

    async def on_user_turn_completed(self, turn_ctx, new_message) -> None:
        """Foreground-агент: как только собеседник договорил, мгновенно (без LLM)
        проигрываем короткий филлер, пока background-конвейер (Sonnet) генерит
        реальный ответ. Это заполняет паузу и звучит по-человечески. Филлер
        прерываемый и НЕ идёт в chat_ctx → BA его не дублирует.

        По умолчанию ВЫКЛ (VOICE_FILLER=on включает): при стриминге BA ответ и так
        начинает звучать быстро, а отдельный филлер давал ощущение дублирования.
        Пропускаем на отказе (там сразу прощание) и на пустой реплике."""
        if os.environ.get("VOICE_FILLER", "off").strip().lower() not in ("on", "1", "true"):
            return
        text = (getattr(new_message, "text_content", "") or "").strip()
        if not text or compliance.detect_refusal(text):
            return
        try:
            self.session.say(objections.foreground_filler(text), add_to_chat_ctx=False)
        except Exception:  # noqa: BLE001 — filler не критичен, не валим ход
            pass


def _env_float(name: str) -> float | None:
    """Парсит float из окружения; пустое/некорректное → None (берётся дефолт плагина)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        print(f"⚠ {name}={raw!r} — не число, использую значение по умолчанию.")
        return None


def _make_tts() -> inworld.TTS:
    # text_normalization=ON: разворачивать числа/даты/сокращения в слова
    # («116» → «сто шестнадцать», «25 лет» → словами) — иначе TTS их коверкает.
    kwargs: dict = {
        "language": os.environ.get("INWORLD_LANG", "ru-RU"),
        "text_normalization": "ON",
        # КЛЮЧЕВОЕ для латентности: дефолты плагина (120 симв / 3000 мс) заставляют
        # стриминговый TTS ждать до 3 с перед синтезом первого куска. Снижаем —
        # синтез стартует почти сразу (первый звук на ~2 с раньше).
        "buffer_char_threshold": int(_env_float("INWORLD_BUFFER_CHARS") or 30),
        "max_buffer_delay_ms": int(_env_float("INWORLD_BUFFER_DELAY_MS") or 120),
    }
    if os.environ.get("INWORLD_VOICE"):
        kwargs["voice"] = os.environ["INWORLD_VOICE"]
    if os.environ.get("INWORLD_TTS_MODEL"):
        kwargs["model"] = os.environ["INWORLD_TTS_MODEL"]
    # audioConfig.speakingRate → speaking_rate [0.5, 1.5]; temperature (0, 2].
    # Дефолт 0.9 (чуть медленнее обычного) — на тесте секретарь жаловался, что
    # агент «тараторит». Переопределяется INWORLD_SPEAKING_RATE.
    rate = _env_float("INWORLD_SPEAKING_RATE")
    kwargs["speaking_rate"] = rate if rate is not None else 0.9
    temp = _env_float("INWORLD_TEMPERATURE")
    if temp is not None:
        kwargs["temperature"] = temp
    return inworld.TTS(**kwargs)


async def entrypoint(ctx: JobContext) -> None:
    state = ConversationState(
        mode=os.environ.get("CHEL_MODE", "A"),  # type: ignore[arg-type]
        phone=os.environ.get("CHEL_PHONE", "+7-LOCAL-TEST"),
        company=os.environ.get("CHEL_COMPANY", "Тестовый рудник"),
        target_role=os.environ.get("CHEL_ROLE") or None,
        lpr_name_hint=os.environ.get("CHEL_LPR_NAME") or None,
        agent_name=os.environ.get("CHEL_AGENT_NAME") or "Олег",
        voice_mode=True,  # короткие реплики + низкая латентность (см. pipeline)
    )
    text_llm = make_llm()
    # Голос: модель реплик переключается VOICE_DIALOGUE_MODEL:
    #   пусто / "sonnet" → Sonnet из make_llm() (ПО УМОЛЧАНИЮ — Haiku на тесте
    #                      выдавал несвязные фразы, секретарь не понимал);
    #   "haiku"          → быстрая Haiku (postprocess_model текущего бэкенда);
    #   <слаг модели>    → использовать его как есть.
    voice_model = os.environ.get("VOICE_DIALOGUE_MODEL", "").strip()
    if voice_model.lower() == "haiku":
        text_llm.dialogue_model = text_llm.postprocess_model
    elif voice_model and voice_model.lower() != "sonnet":
        text_llm.dialogue_model = voice_model
    print(f"[voice] модель реплик: {text_llm.dialogue_model}")

    # AEC-warmup отключает прерывания на первые N c — из-за дефолта 3 c секретарь
    # не мог перебить приветствие. Короче (гарнитура → эхо мало). CHEL_AEC_WARMUP.
    aec_warmup = _env_float("CHEL_AEC_WARMUP")
    if aec_warmup is None:
        aec_warmup = 0.5

    session = AgentSession(
        stt=deepgram.STT(
            model=os.environ.get("DEEPGRAM_MODEL", "nova-3"),
            language=os.environ.get("DEEPGRAM_LANG", "multi"),
        ),
        llm=ChelLLM(state, text_llm),
        tts=_make_tts(),
        # min_silence_duration 0.5: ловит конец реплики, но не дробит на фрагменты.
        vad=silero.VAD.load(min_silence_duration=0.5),
        aec_warmup_duration=aec_warmup,
        # turn_handling задаём ЗДЕСЬ (на сессии) — preemptive читается из опций
        # сессии, не агента.
        turn_handling={
            # Локальная VAD-детекция прерываний (без облачного adaptive → без 401).
            # min_words=1, min_duration=0.5: секретарь может ПЕРЕБИТЬ агента и быть
            # услышанным (иначе его реплики поверх речи агента терялись — «нужно
            # повторять»). 0.5 c отсекает короткие щелчки/эхо.
            "interruption": {"mode": "vad", "min_words": 1, "min_duration": 0.5},
            # 0.7 c тишины: НЕ дробить реплику на фрагменты STT (иначе на каждый
            # кусок генерится отдельный ответ → дубли/«фантомные» реплики).
            "endpointing": {"min_delay": 0.7},
            # Выключаем спекулятивную генерацию — главный источник дублей и
            # ответов, которые не были слышны (генерились на полу-реплику).
            "preemptive_generation": {"enabled": False},
        },
    )

    async def _finalize() -> None:
        """Постобработка на завершении звонка (тот же extractor, что в фазе 1)."""
        if not state.transcript:
            return
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, extractor.extract, state, text_llm)
        path = await loop.run_in_executor(None, extractor.save, state, result)
        print("\n" + extractor.report(result))
        print(f"\n✓ Результат сохранён: {path}")

    ctx.add_shutdown_callback(_finalize)

    # --- Аварийный выход при недоступности STT ---
    # Если провайдер распознавания отваливается (Deepgram рвёт WS, нет транскриптов),
    # агент озвучивает проблему со связью и кладёт трубку. Два триггера:
    #   1) счётчик ошибок STT >= STT_MAX_ERRORS (или неrecoverable-ошибка);
    #   2) watchdog: была ошибка STT и нет распознавания > STT_SILENCE_TIMEOUT с.
    # Счётчик/таймер сбрасываются на каждом успешном транскрипте — одиночный
    # восстановившийся обрыв не считается фатальным.
    max_stt_errors = int(_env_float("STT_MAX_ERRORS") or 2)
    stt_silence_timeout = _env_float("STT_SILENCE_TIMEOUT") or 18.0
    fault = {"stt_errors": 0, "hanging_up": False, "last_rx": time.monotonic()}

    @session.on("user_input_transcribed")
    def _on_transcript(ev) -> None:  # noqa: ANN001
        if getattr(ev, "is_final", False):
            fault["stt_errors"] = 0          # STT снова работает
            fault["last_rx"] = time.monotonic()

    async def _hangup(reason: str) -> None:
        state.refused = True
        state.finished = True
        try:
            handle = session.say(
                "Извините, я вас совсем не слышу — видимо, плохая связь. "
                "Перезвоню позже. Всего доброго!",
                add_to_chat_ctx=False,
            )
            await handle.wait_for_playout()
        except Exception:  # noqa: BLE001 — даже если TTS не сработал, всё равно выходим
            pass
        ctx.shutdown(reason=reason)

    def _trigger_hangup(why: str) -> None:
        if fault["hanging_up"]:
            return
        fault["hanging_up"] = True
        print(f"⚠ {why} — кладу трубку.")
        asyncio.create_task(_hangup("STT unavailable"))

    @session.on("error")
    def _on_error(ev) -> None:  # noqa: ANN001
        err = getattr(ev, "error", None)
        if not isinstance(err, STTError):
            return
        fault["stt_errors"] += 1
        if not getattr(err, "recoverable", True) or fault["stt_errors"] >= max_stt_errors:
            _trigger_hangup(f"STT недоступен ({fault['stt_errors']} ошибок)")

    async def _stt_watchdog() -> None:
        """Если STT уже ошибался и давно нет распознавания — связь мертва, выходим."""
        while not fault["hanging_up"]:
            await asyncio.sleep(3)
            if (fault["stt_errors"] >= 1
                    and time.monotonic() - fault["last_rx"] > stt_silence_timeout):
                _trigger_hangup("STT молчит (нет распознавания)")

    async def _end_call() -> None:
        """Штатное завершение: агент уже произнёс прощание — кладём трубку."""
        await asyncio.sleep(0.4)  # дать последней фразе долиться
        ctx.shutdown(reason="conversation finished")

    @session.on("agent_state_changed")
    def _on_agent_state(ev) -> None:  # noqa: ANN001
        # Разговор помечен завершённым (отказ или стадия WRAP) и агент договорил
        # (перешёл из speaking в listening) → завершаем программу.
        if (state.finished and not fault["hanging_up"]
                and getattr(ev, "new_state", "") == "listening"):
            fault["hanging_up"] = True
            print("✓ Разговор завершён — кладу трубку.")
            asyncio.create_task(_end_call())

    agent = ChelAgent(state)
    # Предзагрузка фонового шума ДО звонка (Inworld TTS: 24кГц моно). Если не
    # удалось — отключаем фон, чтобы он не влиял на живой аудио-путь.
    if agent._ambience is not None:
        try:
            await agent._ambience.preload(24000, 1)
        except Exception as e:  # noqa: BLE001
            print(f"⚠ Фон отключён (ошибка загрузки): {e}")
            agent._ambience = None

    await ctx.connect()
    await session.start(agent=agent, room=ctx.room)
    asyncio.create_task(_stt_watchdog())  # сторож недоступности STT

    # Открытие. Для сценария A приветствие фиксированное — произносим его канонной
    # фразой (мгновенно, без 8-секундной генерации LLM, и одинаково каждый раз;
    # прерывается после короткого AEC-warmup). Дальше диалог ведёт LLM-конвейер.
    if state.mode == "A":
        greeting = (
            f"Добрый день! Меня зовут {state.agent_name}, Челябинский завод "
            f"электрооборудования. Соедините, пожалуйста, с главным энергетиком "
            f"— вопрос по сроку службы ваших подстанций."
        )
        state.add("agent", greeting)  # в контекст нашего конвейера
        session.say(greeting, add_to_chat_ctx=True)
    else:
        await session.generate_reply()  # сценарий B: открытие импровизирует LLM


def main() -> None:
    load_dotenv(REPO / ".env")
    # Локальный console-режим (unregistered) к LiveKit-серверу не подключается,
    # но воркер на старте требует непустые LIVEKIT_*. Подставляем заглушки, если
    # они пусты/не заданы, — для console этого достаточно (реального коннекта нет).
    for name, placeholder in (
        ("LIVEKIT_URL", "ws://localhost:7880"),
        ("LIVEKIT_API_KEY", "devkey"),
        ("LIVEKIT_API_SECRET", "devsecret-console-local-padding-0123456789abcdef"),
    ):
        if not os.environ.get(name):
            os.environ[name] = placeholder
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))


if __name__ == "__main__":
    main()
