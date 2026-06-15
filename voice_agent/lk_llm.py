"""Адаптер: наш текстовый конвейер как LiveKit `llm.LLM` (фаза 2, голос).

LiveKit `AgentSession` требует объект `llm.LLM`. Здесь мы оборачиваем наш конвейер
(director → LLM → compliance) в `llm.LLM`/`LLMStream`, чтобы режиссура/УКС/комплаенс
работали и в голосе. Источник правды — `ConversationState`.

СТРИМИНГ (фоновый/background-агент): ответ модели стримится пофразово в TTS, чтобы
первый звук шёл максимально быстро (латентность Sonnet/llama прячется). Комплаенс
сохраняется: `ComplianceGuard` детерминированный (regex/ключевые слова), поэтому
каждое ЗАКОНЧЕННОЕ предложение проверяется ДО отправки в TTS — без обхода guard.
"""

from __future__ import annotations

import asyncio
import logging
import re

from livekit.agents import llm, utils

logger = logging.getLogger("chel.voice")

from . import compliance
from .director import DialogueDirector
from .pipeline import fallback_reply, generate_reply, prepare_turn
from .providers import LLMProvider
from .state import ConversationState

# Граница предложения: знак конца + пробел (на лету не используем '$', чтобы не
# принять незаконченный хвост буфера за предложение — хвост сбрасываем в конце).
_SENT_BOUNDARY = re.compile(r"[.!?…]+\s")


class ChelLLM(llm.LLM):
    """LiveKit-LLM поверх нашего конвейера сценария A/B."""

    def __init__(self, state: ConversationState, text_llm: LLMProvider) -> None:
        super().__init__()
        self._state = state
        self._text_llm = text_llm
        self._last_user: str | None = None  # дедуп пользовательских ходов

    @property
    def model(self) -> str:
        return "chel-pipeline"

    def chat(self, *, chat_ctx, tools=None, conn_options=None, **kwargs):
        if conn_options is None:
            from livekit.agents import APIConnectOptions
            conn_options = APIConnectOptions()
        return _ChelStream(self, chat_ctx=chat_ctx, tools=tools or [],
                           conn_options=conn_options)


class _ChelStream(llm.LLMStream):
    def __init__(self, chel: ChelLLM, *, chat_ctx, tools, conn_options) -> None:
        super().__init__(chel, chat_ctx=chat_ctx, tools=tools, conn_options=conn_options)
        self._chel = chel
        self._cctx = chat_ctx

    def _emit(self, text: str) -> None:
        self._event_ch.send_nowait(
            llm.ChatChunk(
                id=utils.shortuuid(),
                delta=llm.ChoiceDelta(role="assistant", content=text),
            )
        )

    def _latest_user_text(self) -> str:
        for item in reversed(self._cctx.items):
            if getattr(item, "role", None) == "user":
                return (getattr(item, "text_content", "") or "").strip()
        return ""

    async def _run(self) -> None:
        st = self._chel._state

        # 1) Принять ход собеседника (sync, дёшево — без сети).
        user_text = self._latest_user_text()
        if user_text and user_text != self._chel._last_user:
            self._chel._last_user = user_text
            st.add("interlocutor", user_text)
            if compliance.detect_refusal(user_text):
                st.refused = True
            DialogueDirector(st).advance(user_text)

        # 2) Отказ → короткое прощание, без обращения к LLM.
        if st.refused:
            st.finished = True
            reply = "Понял, извините за беспокойство. Всего доброго!"
            st.add("agent", reply)
            self._emit(reply)
            return

        # 3) Подготовка хода (sync) и стриминг ответа в TTS.
        system, messages, max_tokens = prepare_turn(st, self._chel._text_llm)
        full = await self._stream_reply(system, messages, max_tokens)

        # 4) Стрим пуст/упал → нестриминговый путь (с полным guard); если и он
        #    падает (напр. неверный слаг модели) — безопасная статичная фраза,
        #    чтобы сессия не рухнула и агент хоть что-то ответил.
        if not full:
            loop = asyncio.get_running_loop()
            try:
                full = await loop.run_in_executor(
                    None, lambda: generate_reply(st, self._chel._text_llm))
            except Exception as e:  # noqa: BLE001
                logger.error("LLM недоступна (проверь VOICE_DIALOGUE_MODEL / слаг): %s", e)
                full = ""
            if not full:
                full = fallback_reply(st)
            self._emit(full)

        st.add("agent", full)

    async def _stream_reply(self, system: str, messages: list[dict], max_tokens: int) -> str:
        """Стримит токены LLM, режет на предложения, прогоняет каждое через guard и
        эмитит одобренные → TTS озвучивает инкрементально. Возвращает одобренный
        текст или '' (ошибка/пусто/всё отсеяно) — тогда вызывающий делает fallback."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def worker() -> None:
            buf = ""
            try:
                for piece in self._chel._text_llm.complete_stream(
                        system, messages, max_tokens=max_tokens, temperature=0.7):
                    buf += piece
                    while True:
                        m = _SENT_BOUNDARY.search(buf)
                        if not m:
                            break
                        sent = buf[:m.end()].strip()
                        buf = buf[m.end():]
                        if sent:
                            loop.call_soon_threadsafe(q.put_nowait, ("s", sent))
                tail = buf.strip()
                if tail:
                    loop.call_soon_threadsafe(q.put_nowait, ("s", tail))
            except Exception as e:  # noqa: BLE001 — пробросим как сигнал к fallback
                loop.call_soon_threadsafe(q.put_nowait, ("e", str(e)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("d", None))

        fut = loop.run_in_executor(None, worker)
        parts: list[str] = []
        error = False
        while True:
            kind, val = await q.get()
            if kind == "d":
                break
            if kind == "e":
                error = True
                logger.error("Стрим LLM прервался (проверь VOICE_DIALOGUE_MODEL): %s", val)
                continue
            # kind == "s": законченное предложение — проверяем guard ДО TTS.
            if val and compliance.check_agent_reply(val).ok:
                parts.append(val)
                self._emit(val + " ")
        await fut

        full = " ".join(parts).strip()
        if error and not full:
            return ""
        return full
