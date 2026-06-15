"""Провайдеры STT/LLM/TTS за интерфейсами (требование N5 — заменяемость).

Фаза 1 использует только LLMProvider. Бэкенд выбирается env-переменной
`LLM_BACKEND` (anthropic | openrouter) через фабрику `make_llm()`. STT/TTS —
протоколы-заглушки для фазы 2 (LiveKit + Deepgram + Inworld).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(rel_path: str) -> str:
    """Загрузить промпт из prompts/ (например, 'qualify_system.md')."""
    return (PROMPTS_DIR / rel_path).read_text(encoding="utf-8")


def _require_key(env_name: str, api_key: str | None) -> str:
    """Вернуть ключ или бросить понятную ошибку (пусто/плейсхолдер/нет в .env)."""
    key = api_key or os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"{env_name} не задан. Впишите реальный ключ в .env "
            f"(скопирован из .env.example?) и запустите снова."
        )
    if "..." in key or key.strip() in ("sk-or-", "sk-ant-"):
        raise RuntimeError(
            f"{env_name} содержит значение-плейсхолдер ('{key[:8]}...'). "
            f"Замените его на реальный ключ в .env."
        )
    return key


@runtime_checkable
class LLMProvider(Protocol):
    # Имена моделей: для реплик и для постобработки (extractor).
    dialogue_model: str
    postprocess_model: str

    def complete(self, system: str, messages: list[dict], *, model: str | None = None,
                 max_tokens: int = 400, temperature: float = 0.7) -> str:
        ...

    def complete_stream(self, system: str, messages: list[dict], *, model: str | None = None,
                        max_tokens: int = 400, temperature: float = 0.7) -> Iterator[str]:
        ...


class AnthropicLLM:
    """Реализация LLMProvider на Anthropic SDK."""

    def __init__(self, api_key: str | None = None,
                 dialogue_model: str | None = None,
                 postprocess_model: str | None = None) -> None:
        import anthropic  # ленивый импорт, чтобы пакет грузился без ключа

        self._client = anthropic.Anthropic(api_key=_require_key("ANTHROPIC_API_KEY", api_key))
        self.dialogue_model = dialogue_model or os.environ.get(
            "LLM_DIALOGUE_MODEL", "claude-sonnet-4-6")
        self.postprocess_model = postprocess_model or os.environ.get(
            "LLM_POSTPROCESS_MODEL", "claude-haiku-4-5-20251001")

    def complete(self, system: str, messages: list[dict], *, model: str | None = None,
                 max_tokens: int = 400, temperature: float = 0.7) -> str:
        resp = self._client.messages.create(
            model=model or self.dialogue_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages or [{"role": "user", "content": "(начни разговор)"}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "".join(parts).strip()

    def complete_stream(self, system: str, messages: list[dict], *, model: str | None = None,
                        max_tokens: int = 400, temperature: float = 0.7) -> Iterator[str]:
        with self._client.messages.stream(
            model=model or self.dialogue_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages or [{"role": "user", "content": "(начни разговор)"}],
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text


class OpenRouterLLM:
    """Реализация LLMProvider через OpenRouter (OpenAI-совместимый API).

    Один шлюз к моделям Anthropic/OpenAI/Google и др. Слаги моделей — в формате
    OpenRouter ('anthropic/claude-sonnet-4.6'), выбираются на openrouter.ai/models.
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str | None = None,
                 dialogue_model: str | None = None,
                 postprocess_model: str | None = None,
                 base_url: str | None = None) -> None:
        from openai import OpenAI  # ленивый импорт

        self._client = OpenAI(
            api_key=_require_key("OPENROUTER_API_KEY", api_key),
            base_url=base_url or os.environ.get("OPENROUTER_BASE_URL", self.BASE_URL),
        )
        self.dialogue_model = dialogue_model or os.environ.get(
            "LLM_DIALOGUE_MODEL", "anthropic/claude-sonnet-4.6")
        self.postprocess_model = postprocess_model or os.environ.get(
            "LLM_POSTPROCESS_MODEL", "anthropic/claude-3.5-haiku")
        # Необязательные заголовки OpenRouter для атрибуции (рейтинг приложений).
        self._extra_headers = {
            k: v for k, v in {
                "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", ""),
                "X-Title": os.environ.get("OPENROUTER_APP_NAME", "chel-voice-agent"),
            }.items() if v
        }

    def complete(self, system: str, messages: list[dict], *, model: str | None = None,
                 max_tokens: int = 400, temperature: float = 0.7) -> str:
        # В OpenAI-формате системный промпт — это первое сообщение роли 'system'.
        msgs = [{"role": "system", "content": system}, *(messages or [])]
        resp = self._client.chat.completions.create(
            model=model or self.dialogue_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=msgs,
            extra_headers=self._extra_headers or None,
        )
        return (resp.choices[0].message.content or "").strip()

    def complete_stream(self, system: str, messages: list[dict], *, model: str | None = None,
                        max_tokens: int = 400, temperature: float = 0.7) -> Iterator[str]:
        msgs = [{"role": "system", "content": system}, *(messages or [])]
        stream = self._client.chat.completions.create(
            model=model or self.dialogue_model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=msgs,
            extra_headers=self._extra_headers or None,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def make_llm() -> LLMProvider:
    """Фабрика LLM-провайдера по env `LLM_BACKEND` (anthropic | openrouter)."""
    backend = os.environ.get("LLM_BACKEND", "anthropic").lower()
    if backend == "openrouter":
        return OpenRouterLLM()
    if backend == "anthropic":
        return AnthropicLLM()
    raise ValueError(f"Неизвестный LLM_BACKEND={backend!r} (ожидается anthropic|openrouter)")


# --- Заглушки для фазы 2 (голос) ---

@runtime_checkable
class STTProvider(Protocol):
    def transcribe(self, audio: bytes) -> str: ...


@runtime_checkable
class TTSProvider(Protocol):
    def synthesize(self, text: str) -> bytes: ...
