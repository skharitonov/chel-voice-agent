"""Смоук-тест LLM-провайдера: отправляет один минимальный запрос.

Проверяет, что выбранный бэкенд (LLM_BACKEND) и ключ настроены и доступны.

    python scripts/test_llm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    # Грузим .env из корня репозитория независимо от текущего каталога запуска.
    load_dotenv(REPO_ROOT / ".env")

    import os

    from voice_agent.providers import make_llm

    backend = os.environ.get("LLM_BACKEND", "anthropic")
    print(f"LLM_BACKEND = {backend}")

    try:
        llm = make_llm()
    except Exception as e:  # noqa: BLE001
        print(f"✗ Не удалось создать провайдер: {e}")
        return 1

    print(f"dialogue_model    = {llm.dialogue_model}")
    print(f"postprocess_model = {llm.postprocess_model}")
    print("Отправляю тестовый запрос…")

    try:
        reply = llm.complete(
            system="Ты лаконичный ассистент. Отвечай одним коротким предложением по-русски.",
            messages=[{"role": "user", "content": "Скажи, что связь работает."}],
            max_tokens=50,
            temperature=0,
        )
    except Exception as e:  # noqa: BLE001
        print(f"✗ Запрос не прошёл: {type(e).__name__}: {e}")
        return 1

    print(f"✓ Ответ модели: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
