"""Состояние разговора и Pydantic-модели результатов.

Общий `ConversationState` — единый источник правды, который читают и пишут все
супервайзеры (director, objections, compliance, extractor). Схема результата
сценария A соответствует скрипту 2026-06-11-epb-call-script.md §6; сценария B —
PRD §3.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

Mode = Literal["A", "B"]


class Turn(BaseModel):
    """Одна реплика в транскрипте."""

    role: Literal["agent", "interlocutor", "system"]
    text: str
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class StageA(str, Enum):
    """Стадии сценария A v2 (две фазы: секретарь -> ЛПР). Скрипт §2.4."""

    GATEKEEPER_INTRO = "gatekeeper_intro"          # открытие секретарю (§3)
    GATEKEEPER_OBJECTION = "gatekeeper_objection"  # цикл УКС (повторяемый)
    GATEKEEPER_EXTRACT = "gatekeeper_extract"      # тупик: извлечение данных
    LPR_INTRO = "lpr_intro"                        # вход к ЛПР (§5.1)
    LPR_QUALIFY_1 = "lpr_qualify_1"                # 5 вопросов (§5.2)
    LPR_QUALIFY_2 = "lpr_qualify_2"
    LPR_QUALIFY_3 = "lpr_qualify_3"
    LPR_QUALIFY_4 = "lpr_qualify_4"
    LPR_QUALIFY_5 = "lpr_qualify_5"
    LPR_WRAP = "lpr_wrap"                          # следующий шаг + завершение (§5.3)
    WRAP = "wrap"                                  # финальное завершение (любая фаза)


class StageB(str, Enum):
    INTRO = "intro"            # M1: выбить ФИО
    ANCHOR = "anchor"          # M1: номер-якорь
    ESCALATE = "escalate"      # M2: добавочный -> мобильный
    DETOUR = "detour"          # M3: через другой отдел
    WRAP = "wrap"


class ConversationState(BaseModel):
    """Состояние одного звонка."""

    mode: Mode
    phone: str
    company: Optional[str] = None
    target_role: Optional[str] = None   # для сценария B
    lpr_name_hint: Optional[str] = None  # ФИО из ОСИНТ -> адресное открытие (A, --lpr-name)
    agent_name: str = "Олег"             # имя агента — постоянное (а не разное каждый раз)

    stage: str = ""                    # значение StageA/StageB
    tactic: str = ""                   # человекочитаемая тактика для директивы LLM
    transcript: list[Turn] = Field(default_factory=list)
    facts: dict[str, str] = Field(default_factory=dict)  # накопленные факты

    # Сценарий A: служебные счётчики/флаги (скрипт §2).
    objection_rounds: int = 0          # сколько кругов возражений секретаря пройдено
    extract_rounds: int = 0            # ходов на стадии извлечения (добор ФИО + переключение)
    uks_triggered: list[str] = Field(default_factory=list)  # какие УКС сработали
    secretary_name: Optional[str] = None
    binary_filter: str = "unknown"     # underground_yes | underground_no | unknown
    reached_lpr: bool = False          # произошло ли соединение с ЛПР

    finished: bool = False
    refused: bool = False              # собеседник отказался / просил не звонить
    voice_mode: bool = False           # голос (фаза 2): короткие реплики, низкая латентность

    def add(self, role: str, text: str) -> None:
        self.transcript.append(Turn(role=role, text=text))  # type: ignore[arg-type]

    def history_for_llm(self) -> list[dict]:
        """Транскрипт в формате messages для LLM (agent=assistant)."""
        out: list[dict] = []
        for t in self.transcript:
            if t.role == "agent":
                out.append({"role": "assistant", "content": t.text})
            elif t.role == "interlocutor":
                out.append({"role": "user", "content": t.text})
        return out


# --- Результаты ---


# Сценарий A v2 (скрипт §6)

class Park(BaseModel):
    total: int = 0
    types: list[str] = Field(default_factory=list)   # КТПВШ | КТПВМ | КТПРН
    expired_or_on_epb: int = 0                        # сколько за сроком / на продлении ЭПБ
    last_replacement_year: Optional[int] = None


class Purchase(BaseModel):
    plan: str = ""
    channel: str = ""   # тендер | прямой договор | неизвестно


# Допустимые значения outcome (скрипт §6); храним как str ради устойчивости
# к ответам LLM, нормализуем в extractor.
OUTCOMES = (
    "connected_lpr", "data_extracted", "callback_scheduled",
    "not_qualified", "refused",
)


class QualifyResult(BaseModel):
    company: str = ""
    phone: str = ""
    outcome: str = "not_qualified"   # обязателен и непуст (Правило №1)
    qualified: bool = False
    binary_filter: str = "unknown"   # underground_yes | underground_no | unknown
    lpr_name: str = ""
    lpr_role: str = ""
    lpr_email: str = ""
    lpr_phone_ext: str = ""
    park: Park = Field(default_factory=Park)
    purchase: Purchase = Field(default_factory=Purchase)
    decision_chain: list[str] = Field(default_factory=list)
    callback_at: Optional[str] = None  # YYYY-MM-DD
    secretary_name: str = ""
    uks_triggered: list[str] = Field(default_factory=list)
    notes: str = ""


# Сценарий B (PRD §3)

class LprResult(BaseModel):
    company: str = ""
    phone: str = ""
    target_role: str = ""
    lpr_name: str = ""
    lpr_mobile: str = ""
    method_used: str = ""  # "1" | "2" | "3"
    notes: str = ""
