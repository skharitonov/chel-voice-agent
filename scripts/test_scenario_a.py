"""Детерминированные тесты сценария A v2 (без сети).

Покрывают критерии приёмки spec §3, проверяемые без живого LLM:
- ComplianceGuard блокирует фабрикации (критерий 2);
- классификатор УКС-1…16;
- переходы Director (3 круга возражений -> extract; соединение -> ЛПР;
  бинарный фильтр; цепочка 5 вопросов);
- нормализация outcome (Правило №1);
- полный конвейер text_sim с fake LLM -> валидный JSON с непустым outcome.

Запуск:  .venv/Scripts/python.exe scripts/test_scenario_a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from voice_agent import compliance, extractor, objections, text_sim  # noqa: E402
from voice_agent.director import DialogueDirector  # noqa: E402
from voice_agent.state import ConversationState, QualifyResult, StageA  # noqa: E402

_passed = 0


def check(cond: bool, msg: str) -> None:
    global _passed
    assert cond, f"FAIL: {msg}"
    _passed += 1


def test_compliance_blocks_fabrications() -> None:
    blocked = [
        "Мы договаривались о звонке на той неделе",
        "Он ждёт моего звонка",
        "У вас предписание Ростехнадзора",
        "К вам придёт проверка",
        "Ваши подстанции просрочены",          # безусловно -> блок
        "Я ваш поставщик",
        "Эксплуатация ваших подстанций незаконна",
    ]
    for b in blocked:
        check(not compliance.check_agent_reply(b).ok, f"должно блокироваться: {b!r}")

    allowed = [
        "По 116-ФЗ после 25 лет нужна экспертиза промбезопасности",
        "Если у вас есть подстанции старше 25 лет, потребуется ЭПБ",  # условно — ок
        "Соедините, пожалуйста, с главным энергетиком",
        "Нормативный срок службы КТП по ГОСТ — двадцать пять лет",
    ]
    for a in allowed:
        check(compliance.check_agent_reply(a).ok, f"не должно блокироваться: {a!r}")


def test_uks_classifier() -> None:
    cases = {
        "По какому вопросу вы звоните?": "УКС-1",
        "А вы с нами работаете?": "УКС-2",
        "С кем именно вас соединить?": "УКС-3",
        "Он на площадке сейчас": "УКС-4",
        "Пришлите на почту предложение": "УКС-5",
        "Он не хочет общаться": "УКС-6",
        "Нам ничего не нужно": "УКС-7",
        "Он больше не работает": "УКС-8",
        "Его нет на месте": "УКС-9",
        "Мы работаем с другой компанией": "УКС-10",
        "Это решает головная организация": "УКС-11",
        "Мы сами перезвоним": "УКС-12",
        "Спросила у него — сказал нам не надо": "УКС-13",
        "Мне запрещено соединять": "УКС-14",
        "Все на удалёнке": "УКС-15",
        "Вы что-то предлагаете?": "УКС-16",
    }
    for text, exp in cases.items():
        u = objections.classify(text)
        check(u is not None and u.id == exp,
              f"{text!r} -> {(u.id if u else None)} (ожидалось {exp})")

    # Устойчивость к порядку слов / живым формулировкам (из прогонов персон).
    variants = {
        "Соединить напрямую не могу — у нас такой порядок": "УКС-14",
        "Соединить всё равно не могу, правила есть правила": "УКС-14",
        "Направляйте на общую почту info@company.ru": "УКС-5",
        "Нас это не интересует": "УКС-7",
    }
    for text, exp in variants.items():
        u = objections.classify(text)
        check(u is not None and u.id == exp,
              f"[variant] {text!r} -> {(u.id if u else None)} (ожидалось {exp})")


def test_director_three_rounds_to_extract() -> None:
    st = ConversationState(mode="A", phone="+7", company="X")
    d = DialogueDirector(st)
    check(st.stage == StageA.GATEKEEPER_INTRO.value, "старт = gatekeeper_intro")
    d.advance("По какому вопросу?")
    check(st.stage == StageA.GATEKEEPER_OBJECTION.value, "intro -> objection")
    d.advance("Вы с нами работаете?")
    d.advance("Пришлите на почту")
    check(st.stage == StageA.GATEKEEPER_EXTRACT.value,
          f"после 3 кругов -> extract (stage={st.stage})")
    d.advance("Ладно, записывайте почту")
    check(st.stage == StageA.WRAP.value, "extract -> wrap")


def test_director_connect_to_lpr() -> None:
    st = ConversationState(mode="A", phone="+7")
    d = DialogueDirector(st)
    d.advance("Сейчас соединю, минуту")
    check(st.reached_lpr and st.stage == StageA.LPR_INTRO.value,
          f"соединение -> lpr_intro (stage={st.stage}, reached={st.reached_lpr})")


def test_director_binary_filter_no() -> None:
    st = ConversationState(mode="A", phone="+7")
    st.stage = StageA.GATEKEEPER_OBJECTION.value
    st.uks_triggered.append("УКС-7")
    DialogueDirector(st).advance("Нет, у нас только карьер, нет подземной добычи")
    check(st.binary_filter == "underground_no" and st.stage == StageA.WRAP.value,
          f"бинарный фильтр НЕТ -> wrap (bf={st.binary_filter}, stage={st.stage})")


def test_director_lpr_five_questions() -> None:
    st = ConversationState(mode="A", phone="+7")
    st.stage = StageA.LPR_INTRO.value
    d = DialogueDirector(st)
    d.advance("Да, есть старые подстанции")
    expected = [StageA.LPR_QUALIFY_2, StageA.LPR_QUALIFY_3, StageA.LPR_QUALIFY_4,
                StageA.LPR_QUALIFY_5, StageA.LPR_WRAP]
    check(st.stage == StageA.LPR_QUALIFY_1.value, "lpr_intro -> qualify_1")
    for exp in expected:
        d.advance("ответ")
        check(st.stage == exp.value, f"-> {exp.value} (получили {st.stage})")
    d.advance("ответ")
    check(st.stage == StageA.WRAP.value, "lpr_wrap -> wrap")


def test_outcome_normalization() -> None:
    st = ConversationState(mode="A", phone="+7")
    check(extractor._normalize_outcome("connected_lpr", st) == "connected_lpr",
          "валидный outcome сохраняется")
    st.reached_lpr = True
    check(extractor._normalize_outcome(None, st) == "connected_lpr",
          "reached_lpr -> connected_lpr")
    st2 = ConversationState(mode="A", phone="+7", binary_filter="underground_no")
    check(extractor._normalize_outcome("чепуха", st2) == "not_qualified",
          "underground_no -> not_qualified")
    st3 = ConversationState(mode="A", phone="+7", refused=True)
    check(extractor._normalize_outcome(None, st3) == "refused", "refused -> refused")


class _FakeLLM:
    dialogue_model = "fake-d"
    postprocess_model = "fake-p"

    def complete(self, system, messages, *, model=None, max_tokens=400, temperature=0.7):
        if "СТРОГО JSON" in system:
            return (
                '{"company":"Рудник Глубокий","phone":"+70000000000",'
                '"outcome":"connected_lpr","qualified":true,'
                '"binary_filter":"underground_yes","lpr_name":"Петров Сергей Иванович",'
                '"lpr_role":"главный энергетик","lpr_email":"energo@rudnik.ru",'
                '"lpr_phone_ext":"","park":{"total":12,"types":["КТПВШ","КТПВМ"],'
                '"expired_or_on_epb":4,"last_replacement_year":2003},'
                '"purchase":{"plan":"след. бюджетный цикл","channel":"тендер"},'
                '"decision_chain":["главный инженер","ОМТС"],'
                '"callback_at":"2026-09-01","secretary_name":"Ольга","notes":"4 на ЭПБ"}'
            )
        return "Добрый день! Челябинский завод, соедините с главным энергетиком, пожалуйста."


def test_full_pipeline_fake_llm() -> None:
    st = ConversationState(mode="A", phone="+70000000000", company="Рудник Глубокий")
    scripted = [
        "По какому вопросу?",
        "Соединяю, минуту",
        "Да, есть подстанции из девяностых",
        "Штук двенадцать, КТПВШ и КТПВМ",
        "Четыре на продлении через ЭПБ",
        "Планируем в следующем цикле, тендером",
        "Ещё главный инженер и снабжение",
        "Спасибо, до свидания",
    ]
    text_sim.run(st, llm=_FakeLLM(), scripted=scripted, max_turns=12)

    result = extractor.extract(st, _FakeLLM())
    check(isinstance(result, QualifyResult), "результат A = QualifyResult")
    check(result.outcome in ("connected_lpr", "data_extracted", "callback_scheduled",
                             "not_qualified", "refused") and result.outcome != "",
          f"outcome непуст и валиден: {result.outcome}")
    check(result.park.expired_or_on_epb == 4, "park.expired_or_on_epb распарсен")
    check("УКС-1" in st.uks_triggered, f"УКС-1 залогирован: {st.uks_triggered}")
    check(st.reached_lpr, "дошли до ЛПР")


def main() -> int:
    tests = [
        test_compliance_blocks_fabrications,
        test_uks_classifier,
        test_director_three_rounds_to_extract,
        test_director_connect_to_lpr,
        test_director_binary_filter_no,
        test_director_lpr_five_questions,
        test_outcome_normalization,
        test_full_pipeline_fake_llm,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"✓ {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"✗ {t.__name__}: {e}")
    print(f"\nПроверок пройдено: {_passed}; тестов-провалов: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
