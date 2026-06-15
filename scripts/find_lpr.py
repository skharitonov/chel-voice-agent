"""Сценарий B — Поиск ЛПР (метод Соколова). Entrypoint (фаза 1: текстовая симуляция).

Пример:
  python scripts/find_lpr.py --phone "+7XXXXXXXXXX" --role "главный энергетик" --mode B
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
    p = argparse.ArgumentParser(description="Сценарий B — поиск ЛПР (текстовая симуляция)")
    p.add_argument("--phone", required=True, help="Общий телефон компании / приёмной")
    p.add_argument("--role", required=True, help="Должность нужного ЛПР, напр. 'главный энергетик'")
    p.add_argument("--company", default=None, help="Название компании (опционально)")
    p.add_argument("--mode", default="B", choices=["B"], help="Режим (для совместимости)")
    args = p.parse_args()

    state = ConversationState(mode="B", phone=args.phone, company=args.company,
                             target_role=args.role)
    text_sim.run(state)


if __name__ == "__main__":
    main()
