"""Автономный QA машины диалога (сценарий A) — БЕЗ голоса и без живого LLM.

Прогоняет скриптованные диалоги через реальную логику ветвления/завершения
(pipeline.accept_user_turn + director + pipeline.finalize_agent_reply), проверяя:
  - правильные переходы стадий на репликах собеседника;
  - корректное завершение разговора (finished) — не раньше и не позже нужного;
  - отсутствие «обрыва после собственного вопроса» (баг «недослушал имя»).

Реплики агента в реальности генерит LLM; здесь мы задаём их СКРИПТОМ (помечая,
вопрос это или утверждение) — так проверяется именно машина состояний, а не
качество формулировок. Детерминированно, мгновенно, запускается в CI.

    .venv/Scripts/python.exe scripts/qa_dialogues.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from voice_agent.pipeline import accept_user_turn, finalize_agent_reply  # noqa: E402
from voice_agent.state import ConversationState, StageA  # noqa: E402

_fail = 0
_total = 0


def check(cond: bool, msg: str) -> None:
    global _fail, _total
    _total += 1
    if not cond:
        _fail += 1
        print(f"  ✗ {msg}")
    else:
        print(f"  ✓ {msg}")


def new_state() -> ConversationState:
    return ConversationState(mode="A", phone="+7", company="Рудник", agent_name="Олег",
                             voice_mode=True)


def agent(st: ConversationState, reply: str) -> None:
    """Ход агента (скриптовая реплика; '?' в конце = вопрос)."""
    finalize_agent_reply(st, reply)


def user(st: ConversationState, text: str) -> None:
    """Ход собеседника."""
    accept_user_turn(st, text)


# --- Сценарии ---

def sc_connect_to_lpr() -> None:
    """Секретарь сразу соединяет → фаза ЛПР."""
    print("\n[Сценарий] Секретарь соединяет с ЛПР")
    st = new_state()
    agent(st, "Добрый день! Олег, Челябинский завод. Соедините с энергетиком?")
    user(st, "Минуту, соединяю.")
    check(st.stage == StageA.LPR_INTRO.value, "после 'соединяю' → lpr_intro")
    check(st.reached_lpr, "reached_lpr=True")
    check(not st.finished, "разговор НЕ завершён")


def sc_underground_no() -> None:
    """Бинарный фильтр УКС-7: открытая добыча → непрофиль → wrap."""
    print("\n[Сценарий] Непрофильный клиент (нет подземной добычи)")
    st = new_state()
    agent(st, "Добрый день! Соедините с энергетиком — вопрос по сроку службы подстанций.")
    user(st, "А нам ничего не нужно.")           # УКС-7
    agent(st, "Подскажите, у вас подземная добыча ведётся?")
    user(st, "Нет, у нас только карьер, открытая добыча.")
    check(st.binary_filter == "underground_no", "binary_filter=underground_no")
    check(st.stage == StageA.WRAP.value, "непрофиль → wrap")


def sc_three_rounds_then_extract_fio() -> None:
    """3 круга возражений → extract; добор ФИО за 2 хода; НЕ обрыв после вопроса."""
    print("\n[Сценарий] Тупик → извлечение ФИО (главный кейс бага)")
    st = new_state()
    agent(st, "Добрый день! Соедините с энергетиком?")
    user(st, "По какому вопросу?")               # УКС-1
    agent(st, "Вопрос по сроку службы подстанций. Переключите, пожалуйста?")
    user(st, "Вы с нами работаете?")             # round 2
    agent(st, "Мы завод-производитель. Соедините, пожалуйста?")
    user(st, "Пришлите на почту.")              # round 3 → extract
    check(st.stage == StageA.GATEKEEPER_EXTRACT.value, "3 круга → extract")

    # Агент спрашивает имя энергетика (вопрос).
    agent(st, "Отправлю на общую почту. А как зовут вашего главного энергетика?")
    check(not st.finished, "после вопроса об имени НЕ завершено")
    user(st, "Иван.")                            # дал только имя
    check(st.stage == StageA.GATEKEEPER_EXTRACT.value, "ещё в extract (добор)")

    # Ход-реакция: агент добирает отчество/фамилию (вопрос).
    agent(st, "Иван — а отчество и фамилию подскажете?")
    check(not st.finished, "добор ФИО: НЕ завершено")
    user(st, "Иван Петрович Сидоров.")
    # Теперь extract_rounds>=2 → WRAP. Агент просит переключить (вопрос).
    check(st.stage == StageA.WRAP.value, "после 2 ходов extract → wrap")
    agent(st, "Спасибо! Переключите, пожалуйста, на Ивана Петровича?")
    check(not st.finished, "просьба переключить (вопрос) — НЕ завершено")


def sc_no_hangup_right_after_question_on_wrap() -> None:
    """Главный баг: вопрос на стадии WRAP → ответ → НЕ прощаться немедленно."""
    print("\n[Сценарий] Вопрос на WRAP → ответ → ход-реакция (не обрыв)")
    st = new_state()
    st.stage = StageA.WRAP.value
    st.facts["x"] = "y"
    # Агент на WRAP задаёт вопрос (Правило №1).
    agent(st, "Подскажите, на чьё имя письмо — как зовут энергетика?")
    check(not st.finished, "вопрос на WRAP: НЕ завершено")
    user(st, "Иван Петрович.")
    # Ход-реакция (утверждение) — НЕ должен завершить (prev был вопрос).
    agent(st, "Спасибо, помечу для Ивана Петровича.")
    check(not st.finished, "ход-реакция сразу после вопроса: НЕ завершено (ключевой фикс)")
    # Следующий ход — прощание (оба не вопросы) → завершаем.
    user(st, "Хорошо.")
    agent(st, "Всего доброго, до свидания.")
    check(st.finished, "прощание после реакции → завершено")


def sc_refusal() -> None:
    """Явный отказ → завершение."""
    print("\n[Сценарий] Отказ собеседника")
    st = new_state()
    agent(st, "Добрый день! Соедините с энергетиком?")
    user(st, "Не звоните больше сюда.")
    check(st.refused, "refused=True")
    check(st.stage == StageA.WRAP.value, "отказ → wrap")


def main() -> int:
    print("=== Автономный QA машины диалога (сценарий A) ===")
    sc_connect_to_lpr()
    sc_underground_no()
    sc_three_rounds_then_extract_fio()
    sc_no_hangup_right_after_question_on_wrap()
    sc_refusal()
    print(f"\nПроверок: {_total}; провалов: {_fail}")
    return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
