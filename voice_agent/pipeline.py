"""Конвейер одного хода реплики: Director -> LLM -> Guard.

Используется и текстовой симуляцией (фаза 1), и голосовым ядром (фаза 2) — общая
точка, чтобы логика диалога не дублировалась между режимами.
"""

from __future__ import annotations

from . import compliance, objections
from .director import DialogueDirector
from .providers import LLMProvider, load_prompt
from .state import ConversationState

_SYSTEM_FILE = {"A": "qualify_system.md", "B": "find_lpr_system.md"}


def build_system_prompt(state: ConversationState) -> str:
    """Системный промпт = сценарий + общие тактики возражений и комплаенса."""
    parts = [
        load_prompt(_SYSTEM_FILE[state.mode]),
        "\n\n# Тактики возражений\n" + load_prompt("shared/objections.md"),
        "\n\n# Комплаенс\n" + load_prompt("shared/compliance.md"),
        f"\n\n# Контекст звонка\n"
        f"Компания: {state.company or '—'}. Телефон: {state.phone}. "
        + (f"Целевая роль ЛПР: {state.target_role}." if state.mode == "B" else "")
        + (f" Известно ФИО энергетика (адресное открытие): {state.lpr_name_hint}."
           if state.mode == "A" and state.lpr_name_hint else ""),
    ]
    return "".join(parts)


def prepare_turn(state: ConversationState, llm: LLMProvider) -> tuple[str, list[dict], int]:
    """Подготовить ход: директива стадии, возражение, антиповтор, тон, краткость.
    Возвращает (system, messages, max_tokens). Общая часть для обычного и
    стримингового вызова (голос). НЕ зовёт LLM."""
    director = DialogueDirector(state)
    directive = director.directive()

    last_interlocutor = next(
        (t.text for t in reversed(state.transcript) if t.role == "interlocutor"), "")
    uks = objections.classify(last_interlocutor)

    control = f"[СТАДИЯ/ТАКТИКА: {directive}]"
    if uks:
        if uks.id not in state.uks_triggered:
            state.uks_triggered.append(uks.id)
        control += f"\n[ВОЗРАЖЕНИЕ {uks.id}: {uks.hint}]"

    # Антиповтор: не давать перефразировать уже сказанное (борьба с «пластинкой»).
    last_agent = next(
        (t.text for t in reversed(state.transcript) if t.role == "agent"), "")
    if last_agent:
        control += (f"\n[НЕ ПОВТОРЯЙ свою прошлую реплику и её смысл: «{last_agent[:160]}». "
                    f"Скажи что-то новое, продвигающее разговор.]")

    # Тон: если собеседник резок/раздражён — живой, не приторный регистр.
    if state.voice_mode and objections.detect_irritation(last_interlocutor):
        filler = objections.pick_filler(last_interlocutor)
        control += (f"\n[ТОН: собеседник раздражён/резок. Не приторно-корпоративно. "
                    f"Начни с живой вставки вроде «{filler}», ответь предельно коротко "
                    f"и по делу, как застигнутый врасплох живой менеджер.]")

    # Голос: краткость задаём ПРОМПТОМ; max_tokens — лишь потолок-страховка, он
    # не должен резать легитимную короткую реплику (иначе приветствие обрывается).
    max_tokens = 400
    if state.voice_mode:
        stage = state.stage or ""
        if stage == "gatekeeper_intro":
            max_tokens = 110   # приветствие — одна фраза, но не резать на полуслове
        elif stage.startswith("gatekeeper"):
            max_tokens = 75    # ответ секретарю — 1 короткое предложение
        else:
            max_tokens = 160   # фаза ЛПР — допускаем чуть длиннее
        control += ("\n[ГОЛОС: одна мысль, 1–2 коротких предложения, целиком "
                    "законченные. Без перечислений и монолога. Приветствие — одной фразой.]")

    system = build_system_prompt(state)
    messages = state.history_for_llm()
    messages.append({"role": "user", "content": control})
    return system, messages, max_tokens


def fallback_reply(state: ConversationState) -> str:
    """Безопасная заглушка по сценарию (если guard так и не пропустил / стрим пуст)."""
    return (
        "Подскажите, как связаться с вашим главным энергетиком — "
        "по вопросу сроков службы подстанций?"
        if state.mode == "A"
        else "Извините, я по поводу письма для вашего энергетика. Отчество напомните?"
    )


def generate_reply(state: ConversationState, llm: LLMProvider,
                   *, max_regen: int = 1) -> str:
    """Сгенерировать следующую реплику агента (нестриминговый путь, фаза 1/текст)."""
    system, messages, max_tokens = prepare_turn(state, llm)
    reply = llm.complete(system, messages, max_tokens=max_tokens, temperature=0.7)

    # ComplianceGuard: при нарушении — регенерация с подсказкой.
    for _ in range(max_regen):
        verdict = compliance.check_agent_reply(reply)
        if verdict.ok:
            break
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user",
                         "content": f"[GUARD: {compliance.REGENERATION_HINT}]"})
        reply = llm.complete(system, messages, max_tokens=max_tokens, temperature=0.5)
    else:
        if not compliance.check_agent_reply(reply).ok:
            reply = fallback_reply(state)

    return reply
