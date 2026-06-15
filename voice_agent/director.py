"""DialogueDirector — «режиссёр» диалога.

Отслеживает стадию разговора и выбирает тактику, возвращая короткую директиву для
main LLM. Сценарий A v2 — две фазы (секретарь -> ЛПР) со скриптом
2026-06-11-epb-call-script.md; сценарий B — метод Соколова (без изменений).

Логика переходов детерминированная и эвристическая (по ключевым словам реплики
собеседника). Распознавание возражений секретаря делегируется objections.classify.
"""

from __future__ import annotations

from . import objections
from .state import ConversationState, StageA, StageB

# --- Директивы сценария A (текст для main LLM) ---
_A_DIRECTIVES = {
    StageA.GATEKEEPER_INTRO.value:
        "ФАЗА СЕКРЕТАРЬ. Открытие ОДНОЙ короткой фразой по схеме: «Меня зовут <имя>, "
        "Челябинский завод электрооборудования. Соедините с главным энергетиком — "
        "вопрос по сроку службы ваших подстанций». Причину назови ПОЛНОСТЬЮ и ясно "
        "(«вопрос по сроку службы подстанций»), не обрывай до «энергетик по "
        "подстанциям». НЕ перечисляй детали закона/ГОСТ/типы — это для ЛПР.",
    StageA.GATEKEEPER_OBJECTION.value:
        "ФАЗА СЕКРЕТАРЬ. Ответь на возражение по УКС (подсказка [ВОЗРАЖЕНИЕ УКС-N]) "
        "ОДНОЙ фразой. КРИТИЧНО: НЕ повторяй формулировку про подстанции / срок "
        "службы / 116-ФЗ — ты её уже сказал в открытии. Даже если секретарь "
        "переспрашивает «по каким подстанциям / зачем» — НЕ пересказывай закон, "
        "ГОСТ и «25 лет»: ответь «это технический вопрос к энергетику» и снова "
        "коротко попроси соединить (Правило №3).",
    StageA.GATEKEEPER_EXTRACT.value:
        "ФАЗА СЕКРЕТАРЬ, ТУПИК. Соединения нет — извлекай контакт ЛПР. Если назвали "
        "только имя — НЕ прощайся, уточни ОТЧЕСТВО и ФАМИЛИЮ («а как по отчеству, "
        "фамилию подскажете?») и снова попроси ПЕРЕКЛЮЧИТЬ на него. Цель по убыванию: "
        "ФИО+должность ЛПР → переключение → личная почта/добавочный → дата перезвона. "
        "Ни одна ветка не выходит пустой (Правило №1).",
    StageA.LPR_INTRO.value:
        "ФАЗА ЛПР. Вход (скрипт §5.1): поздоровайся, представь ЧЗЭ и крючок ЭПБ/"
        "116-ФЗ/25 лет, спроси, есть ли подстанции, отработавшие срок. Дай ответить, "
        "не монолог.",
    StageA.LPR_QUALIFY_1.value:
        "ФАЗА ЛПР, вопрос 1 (профиль): подземные выработки, подстанции в шахтном "
        "исполнении есть? Нет → непрофильный, заверши. Один вопрос, жди ответа.",
    StageA.LPR_QUALIFY_2.value:
        "ФАЗА ЛПР, вопрос 2 (размер парка): сколько подстанций и какие типы (КТПВШ/"
        "КТПВМ/КТПРН)? Мини-ценность: делаем все три типа. Один вопрос, жди ответа.",
    StageA.LPR_QUALIFY_3.value:
        "ФАЗА ЛПР, вопрос 3 (боль): сколько уже на продлении через ЭПБ? Мини-ценность: "
        "на горизонте 5–7 лет замена часто дешевле продлений. Один вопрос, жди ответа.",
    StageA.LPR_QUALIFY_4.value:
        "ФАЗА ЛПР, вопрос 4 (сроки/план): замена в этом году или след. цикле? Как "
        "закупаете — тендер/прямой договор? Один вопрос, жди ответа.",
    StageA.LPR_QUALIFY_5.value:
        "ФАЗА ЛПР, вопрос 5 (цепочка решения): кто ещё участвует — гл. инженер, "
        "снабжение, проектный институт? Один вопрос, жди ответа.",
    StageA.LPR_WRAP.value:
        "ФАЗА ЛПР, завершение (скрипт §5.3): договорись о следующем шаге (инженер ЧЗЭ "
        "/ ТКП на личную почту / перезвон к дате бюджета), возьми личную почту и дату.",
    StageA.WRAP.value:
        "Заверши разговор вежливо.",
}

_A_QUALIFY = [
    StageA.LPR_QUALIFY_1.value, StageA.LPR_QUALIFY_2.value, StageA.LPR_QUALIFY_3.value,
    StageA.LPR_QUALIFY_4.value, StageA.LPR_QUALIFY_5.value,
]
_A_GATEKEEPER = {
    StageA.GATEKEEPER_INTRO.value, StageA.GATEKEEPER_OBJECTION.value,
    StageA.GATEKEEPER_EXTRACT.value,
}

# --- Директивы сценария B (метод Соколова) ---
_B_DIRECTIVES = {
    StageB.INTRO.value: "Метод 1. Без представления, уверенно: попроси напомнить "
                        "отчество нужного ЛПР ('пишу письмо вашему ...').",
    StageB.ANCHOR.value: "Метод 1. Получив ФИО, назови выдуманный номер-якорь "
                         "('номер не менялся? 9XX...') — пусть исправят и продиктуют реальный.",
    StageB.ESCALATE.value: "Метод 2. Эскалируй: якорь на добавочный; если его нет — "
                           "на личный мобильный; при 'в командировке' — последние две цифры.",
    StageB.DETOUR.value: "Метод 3. Зайди как через другой отдел; на встречные вопросы "
                         "отвечай односложно и сразу возвращай свой вопрос.",
    StageB.WRAP.value: "Поблагодари и заверши разговор.",
}
_B_ORDER = [s.value for s in StageB]


class DialogueDirector:
    def __init__(self, state: ConversationState) -> None:
        self.state = state
        if not state.stage:
            state.stage = (StageA.GATEKEEPER_INTRO if state.mode == "A"
                           else StageB.INTRO).value

    # ------- директива текущей стадии -------
    def directive(self) -> str:
        if self.state.mode == "A":
            d = self._directive_a()
        else:
            d = _B_DIRECTIVES.get(self.state.stage, "")
        self.state.tactic = d
        return d

    def _directive_a(self) -> str:
        s = self.state
        if s.stage == StageA.WRAP.value:
            # Rule №1: если ничего не извлечено и не отказ/непрофиль — последняя попытка.
            if s.refused:
                return "Заверши: извинись за беспокойство и попрощайся словами «До свидания»."
            if s.binary_filter == "underground_no":
                return ("Заверши: поблагодари, отметь непрофильность, попрощайся "
                        "словами «До свидания».")
            if not self._has_extracted_data():
                return ("ТУПИК перед завершением (Правило №1): попроси сейчас хотя бы "
                        "одно — ФИО+должность ЛПР, личную почту, добавочный или дату "
                        "перезвона; затем попрощайся словами «До свидания».")
            return ("Заверши разговор вежливо, подтверди договорённость, попрощайся "
                    "словами «До свидания».")
        return _A_DIRECTIVES.get(s.stage, "")

    def _has_extracted_data(self) -> bool:
        """Грубая проверка по транскрипту: есть ли признаки извлечённых данных."""
        text = " ".join(t.text.lower() for t in self.state.transcript
                        if t.role == "interlocutor")
        markers = ("@", "добавочн", "перезвон", "почт", "отчеств", "зовут",
                   "имя-отчество", "фамилия")
        return any(m in text for m in markers) or self.state.reached_lpr

    # ------- продвижение стадии -------
    def advance(self, last_interlocutor_text: str) -> None:
        s = self.state
        if s.refused:
            s.stage = (StageA.WRAP if s.mode == "A" else StageB.WRAP).value
            return
        if s.mode == "A":
            self._advance_a(last_interlocutor_text or "")
        else:
            self._advance_b((last_interlocutor_text or "").lower())

    def _advance_a(self, text: str) -> None:
        s = self.state
        stage = s.stage

        # Фаза секретаря.
        if stage in _A_GATEKEEPER:
            if objections.is_connect_signal(text):
                s.reached_lpr = True
                s.stage = StageA.LPR_INTRO.value
                return
            # Бинарный фильтр УКС-7.
            if "УКС-7" in s.uks_triggered:
                ans = objections.underground_answer(text)
                if ans != "unknown":
                    s.binary_filter = ans
                    if ans == "underground_no":
                        s.facts["not_qualified_reason"] = "нет подземной добычи"
                        s.stage = StageA.WRAP.value
                        return
            if stage == StageA.GATEKEEPER_INTRO.value:
                s.stage = StageA.GATEKEEPER_OBJECTION.value
                s.objection_rounds += 1
                return
            if stage == StageA.GATEKEEPER_OBJECTION.value:
                s.objection_rounds += 1
                if s.objection_rounds >= 3:          # максимум 3 круга возражений
                    s.stage = StageA.GATEKEEPER_EXTRACT.value
                return
            if stage == StageA.GATEKEEPER_EXTRACT.value:
                # Даём добрать ФИО и попросить переключить (а не свернуться после
                # первого же слова-имени). В WRAP — только после 2 ходов извлечения.
                s.extract_rounds += 1
                if s.extract_rounds >= 2:
                    s.stage = StageA.WRAP.value
                return

        # Фаза ЛПР.
        if stage == StageA.LPR_INTRO.value:
            s.stage = StageA.LPR_QUALIFY_1.value
            return
        if stage in _A_QUALIFY:
            idx = _A_QUALIFY.index(stage)
            s.stage = (_A_QUALIFY[idx + 1] if idx < len(_A_QUALIFY) - 1
                       else StageA.LPR_WRAP.value)
            return
        if stage == StageA.LPR_WRAP.value:
            s.stage = StageA.WRAP.value
            return

    def _advance_b(self, text: str) -> None:
        s = self.state
        order = _B_ORDER
        idx = order.index(s.stage) if s.stage in order else 0
        # Эскалация на M3 при жёстком блоке.
        if any(w in text for w in ("не дам", "не скажу", "кто это", "не положено")):
            s.stage = StageB.DETOUR.value
            return
        # Якорь -> эскалация, если добавочного нет.
        if s.stage == StageB.ANCHOR.value and any(
                w in text for w in ("нет добавочного", "командировк", "не помню")):
            s.stage = StageB.ESCALATE.value
            return
        if text.strip() and idx < len(order) - 1:
            s.stage = order[idx + 1]
