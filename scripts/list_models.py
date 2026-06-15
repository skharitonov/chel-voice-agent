"""Показать модели, доступные на текущем LLM-бэкенде (для VOICE_DIALOGUE_MODEL).

Полезно, когда `chat/completions` отвечает «Invalid model name» — выводит точные
слаги, которые принимает твой ключ/прокси.

    .venv/Scripts/python.exe scripts/list_models.py [фильтр]

Опциональный аргумент — подстрока-фильтр (напр. `llama`, `gemma`, `claude`).
"""

from __future__ import annotations

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


def main() -> int:
    load_dotenv(REPO / ".env")
    flt = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    backend = os.environ.get("LLM_BACKEND", "anthropic").lower()

    if backend == "anthropic":
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        ids = [m.id for m in client.models.list().data]
    else:  # openrouter / OpenAI-совместимый прокси
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        )
        ids = [m.id for m in client.models.list().data]

    ids = sorted(ids)
    if flt:
        ids = [i for i in ids if flt in i.lower()]
    print(f"Бэкенд: {backend}. Моделей: {len(ids)}"
          + (f" (фильтр: {flt!r})" if flt else ""))
    for i in ids:
        print(" ", i)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
