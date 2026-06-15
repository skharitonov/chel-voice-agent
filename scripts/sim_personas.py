"""Автотест сценария A v2 LLM-секретарём (подход autoresearcher, spec §3).

Прогоняет агента против 6 персон секретаря/ЛПР, каждая — отдельная LLM в роли
собеседника. Печатает транскрипт, отчёт и пишет JSON в results/.

ТРЕБУЕТ рабочего LLM-бэкенда (см. .env: LLM_BACKEND + ключ). Без сети не работает —
для офлайн-проверки логики используйте scripts/test_scenario_a.py.

Запуск:   .venv/Scripts/python.exe scripts/sim_personas.py [--persona N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from dotenv import load_dotenv  # noqa: E402

from voice_agent import compliance, extractor  # noqa: E402
from voice_agent.director import DialogueDirector  # noqa: E402
from voice_agent.pipeline import generate_reply  # noqa: E402
from voice_agent.providers import make_llm  # noqa: E402
from voice_agent.state import ConversationState, StageA  # noqa: E402

# 6 персон из spec §3 (таблица).
PERSONAS = [
    {
        "name": "1. Дружелюбный секретарь, сразу соединяет",
        "persona": "Ты — дружелюбный секретарь рудника. После короткого приветствия "
                   "охотно соединяешь с главным энергетиком. Когда соединяешь — скажи "
                   "'Соединяю, минуту' и дальше отвечай уже как главный энергетик "
                   "Сергей Иванович: у вас 12 подстанций КТПВШ/КТПВМ из начала 2000-х, "
                   "4 на продлении через ЭПБ, замена в следующем бюджетном цикле "
                   "тендером, в решении участвуют главный инженер и снабжение.",
        "expect": "connected_lpr, дошли до lpr_wrap",
    },
    {
        "name": "2. Фильтр 'по какому вопросу?' ×2, потом соединяет",
        "persona": "Ты — секретарь. Дважды переспроси 'по какому вопросу?' и 'вы что-то "
                   "предлагаете?', получив вменяемый ответ про сроки службы и закон — "
                   "соединяешь ('Соединяю'). Дальше отвечай как энергетик кратко.",
        "expect": "УКС-1/УКС-16 отработаны, соединение",
    },
    {
        "name": "3. 'Пришлите на почту', имя не называет",
        "persona": "Ты — секретарь. Настойчиво просишь прислать всё на почту, "
                   "соединять отказываешься. Имя энергетика прямо не называешь, но если "
                   "агент убедит (про спам), назовёшь почту energo@company.ru и фамилию "
                   "Иванов. Не груби, не клади трубку.",
        "expect": "data_extracted: почта + ФИО или почта + дата перезвона",
    },
    {
        "name": "4. 'Нам ничего не нужно', подземной добычи нет",
        "persona": "Ты — секретарь карьера (открытая добыча). Говоришь 'нам ничего не "
                   "нужно'. На вопрос про подземную добычу честно отвечаешь, что её нет "
                   "— только карьер, открытым способом.",
        "expect": "not_qualified, бинарный фильтр, разговор короткий",
    },
    {
        "name": "5. 'Запрещено соединять', жёсткий",
        "persona": "Ты — строгий секретарь по инструкции: соединять запрещено. Но не "
                   "грубишь. Если агент попросит — продиктуешь рабочую почту "
                   "energetik@company.ru и подскажешь добавочный 132. Трубку не кладёшь.",
        "expect": "data_extracted или callback_scheduled, без нарушений compliance",
    },
    {
        "name": "6. ЛПР: 'у нас всё новое'",
        "persona": "Ты сразу отвечаешь как главный энергетик (секретарь уже соединил). "
                   "Говоришь, что парк недавно обновлён, последняя замена в 2022 году, "
                   "всё в порядке. Не груб. Готов дать личную почту energo@company.ru "
                   "для справки по срокам.",
        "expect": "not_qualified (отложен), callback_at заполнен, год замены записан",
    },
]


def _secretary_system(persona: str) -> str:
    return (
        "Ты играешь собеседника в РОЛЕВОЙ симуляции холодного звонка (секретарь или "
        "ЛПР предприятия). Отвечай коротко и естественно, по-русски, ОДНОЙ репликой, "
        "без пояснений и без ремарок. Веди себя строго по своей роли:\n\n" + persona +
        "\n\nЕсли разговор логически завершён — попрощайся. Никогда не выходи из роли."
    )


def run_persona(idx: int, persona: dict, max_turns: int = 18) -> None:
    agent_llm = make_llm()
    secretary_llm = make_llm()
    state = ConversationState(mode="A", phone="+70000000000", company="Тестовый рудник")
    sec_system = _secretary_system(persona["persona"])

    print("\n" + "=" * 70)
    print(f"ПЕРСОНА {persona['name']}")
    print(f"Ожидается: {persona['expect']}")
    print("=" * 70)

    agent_text = generate_reply(state, agent_llm)
    state.add("agent", agent_text)
    print(f"АГЕНТ: {agent_text}\n")

    for _ in range(max_turns):
        # Реплика собеседника (LLM-персона).
        sec_msgs = [{"role": "user", "content": agent_text}]
        # Контекст: предыдущие реплики агента как 'user', свои — как 'assistant'.
        history = []
        for t in state.transcript:
            if t.role == "agent":
                history.append({"role": "user", "content": t.text})
            elif t.role == "interlocutor":
                history.append({"role": "assistant", "content": t.text})
        user_text = secretary_llm.complete(sec_system, history or sec_msgs,
                                           temperature=0.8, max_tokens=120)
        print(f"СОБЕСЕДНИК: {user_text}")
        state.add("interlocutor", user_text)

        if compliance.detect_refusal(user_text):
            state.refused = True
            bye = "Понял, извините за беспокойство. Всего доброго!"
            state.add("agent", bye)
            print(f"\nАГЕНТ: {bye}")
            break

        DialogueDirector(state).advance(user_text)
        agent_text = generate_reply(state, agent_llm)
        state.add("agent", agent_text)
        print(f"\nАГЕНТ: {agent_text}\n")

        if state.stage == StageA.WRAP.value:
            break

    print("\n--- Постобработка ---")
    result = extractor.extract(state, agent_llm)
    print(extractor.report(result))
    path = extractor.save(state, result)
    print(f"\n✓ JSON: {path}")
    print(f"  outcome={result.outcome}  uks={state.uks_triggered}  "
          f"binary_filter={result.binary_filter}")


def main() -> int:
    load_dotenv(REPO / ".env")
    ap = argparse.ArgumentParser(description="Прогон персон секретаря (LLM)")
    ap.add_argument("--persona", type=int, default=None,
                    help="Номер персоны 1..6 (по умолчанию — все)")
    args = ap.parse_args()

    chosen = ([PERSONAS[args.persona - 1]] if args.persona else PERSONAS)
    start = (args.persona or 1)
    for i, p in enumerate(chosen, start=start):
        try:
            run_persona(i, p)
        except Exception as e:  # noqa: BLE001
            print(f"\n✗ Персона {i} прервана: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
