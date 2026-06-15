"""Сценарий A — Квалификация лида. Entrypoint (фаза 1: текстовая симуляция).

Пример:
  python scripts/qualify.py --phone "+7XXXXXXXXXX" --company "Рудник Северный" --mode A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from voice_agent import text_sim  # noqa: E402
from voice_agent.state import ConversationState  # noqa: E402


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass
    load_dotenv()
    p = argparse.ArgumentParser(description="Сценарий A — квалификация лида (текстовая симуляция)")
    p.add_argument("--phone", required=True, help="Телефон компании")
    p.add_argument("--company", required=True, help="Название компании")
    p.add_argument("--lpr-name", default=None, dest="lpr_name",
                   help="ФИО ЛПР из ОСИНТ (опц.) — адресное открытие секретарю")
    p.add_argument("--mode", default="A", choices=["A"], help="Режим (для совместимости)")
    args = p.parse_args()

    state = ConversationState(mode="A", phone=args.phone, company=args.company,
                             lpr_name_hint=args.lpr_name)
    text_sim.run(state)


if __name__ == "__main__":
    main()
