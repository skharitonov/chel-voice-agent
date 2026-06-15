"""Текстовая симуляция звонка (фаза 1) — рабочий прототип без аудио.

Тот же конвейер, что и голосовое ядро: Director -> LLM -> Guard, затем постобработка
(extractor + reporter). Реплики собеседника берутся из stdin (интерактивно) либо из
заранее заданного списка (для автотестов/демо).
"""

from __future__ import annotations

import sys

from . import compliance, extractor
from .director import DialogueDirector
from .pipeline import generate_reply
from .providers import LLMProvider, make_llm
from .state import ConversationState


def run(state: ConversationState, llm: LLMProvider | None = None,
        scripted: list[str] | None = None, max_turns: int = 20) -> None:
    """Прогнать диалог.

    scripted: если задан — реплики собеседника берутся отсюда (демо/тест),
    иначе читаются из input(). Когда scripted исчерпан — диалог завершается.
    """
    # Windows-консоль по умолчанию cp1251 — переключаем вывод на UTF-8 для кириллицы.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass

    llm = llm or make_llm()
    scripted_iter = iter(scripted) if scripted is not None else None

    print(f"\n=== Симуляция звонка | режим {state.mode} | {state.company or state.phone} ===")
    if state.mode == "B":
        print(f"    Цель: ЛПР с ролью «{state.target_role}»")
    print("    (Вы играете собеседника. Пустой ввод или 'end' — завершить.)\n")

    # Первая реплика агента.
    agent_text = generate_reply(state, llm)
    state.add("agent", agent_text)
    print(f"АГЕНТ: {agent_text}\n")

    for _ in range(max_turns):
        if state.finished:
            break

        # Реплика собеседника.
        if scripted_iter is not None:
            user_text = next(scripted_iter, None)
            if user_text is None:
                break
            print(f"СОБЕСЕДНИК: {user_text}")
        else:
            try:
                user_text = input("СОБЕСЕДНИК: ").strip()
            except EOFError:
                break
            if user_text.lower() in ("", "end", "конец"):
                break

        state.add("interlocutor", user_text)

        # ComplianceGuard: отказ собеседника -> завершаем.
        if compliance.detect_refusal(user_text):
            state.refused = True
            state.finished = True
            bye = "Понял, извините за беспокойство. Всего доброго!"
            state.add("agent", bye)
            print(f"\nАГЕНТ: {bye}")
            break

        # Продвигаем стадию и генерим ответ.
        DialogueDirector(state).advance(user_text)
        agent_text = generate_reply(state, llm)
        state.add("agent", agent_text)
        print(f"\nАГЕНТ: {agent_text}\n")

    # Постобработка.
    print("\n--- Постобработка транскрипта ---")
    result = extractor.extract(state, llm)
    print(extractor.report(result))
    path = extractor.save(state, result)
    print(f"\n✓ Результат сохранён: {path}")
