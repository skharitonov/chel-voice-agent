"""Бенчмарк скорости моделей на текущем LLM-бэкенде (OpenRouter/прокси).

В API OpenRouter скорости нет — меряем сами: TTFT (время до первого токена) и
пропускную способность (токенов/с) на коротком русском запросе со стримингом.
Для голоса важнее всего TTFT.

    .venv/Scripts/python.exe scripts/bench_models.py [слаг1 слаг2 ...]

Без аргументов берёт модели из `scripts/list_models.py` (что отдаёт прокси).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv  # noqa: E402

_PROMPT = "Ответь одним коротким предложением: чем полезна взрывозащита подстанции?"


def _client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )


def bench(client, model: str) -> tuple[float, float, int, str]:
    """Возвращает (ttft_s, total_s, tokens, текст). При ошибке ttft=-1."""
    t0 = time.perf_counter()
    ttft = -1.0
    text = ""
    n = 0
    try:
        stream = client.chat.completions.create(
            model=model, max_tokens=80, temperature=0.5, stream=True,
            messages=[{"role": "user", "content": _PROMPT}],
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                if ttft < 0:
                    ttft = time.perf_counter() - t0
                text += delta
                n += 1
        return ttft, time.perf_counter() - t0, n, text.strip()
    except Exception as e:  # noqa: BLE001
        return -1.0, time.perf_counter() - t0, 0, f"ОШИБКА: {e}"


def main() -> int:
    load_dotenv(REPO / ".env")
    models = sys.argv[1:]
    if not models:
        client = _client()
        models = sorted(m.id for m in client.models.list().data)
    client = _client()
    print(f"Бенчмарк {len(models)} моделей (стриминг, TTFT = время до 1-го токена):\n")
    rows = []
    for m in models:
        ttft, total, n, text = bench(client, m)
        rows.append((m, ttft, total, n))
        if ttft < 0:
            print(f"  ✗ {m}: {text}")
        else:
            tps = n / total if total else 0
            print(f"  {m}")
            print(f"      TTFT {ttft*1000:6.0f} мс | всего {total:5.2f} с | ~{tps:4.1f} tok/s")
            print(f"      «{text[:90]}»")
    ok = [r for r in rows if r[1] >= 0]
    if ok:
        print("\nПо TTFT (быстрее — лучше для голоса):")
        for m, ttft, total, n in sorted(ok, key=lambda r: r[1]):
            print(f"  {ttft*1000:6.0f} мс  {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
