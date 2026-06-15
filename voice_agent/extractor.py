"""JSONExtractor + Reporter — постобработка транскрипта.

После завершения разговора прогоняет полный транскрипт через дешёвую модель
(Haiku), извлекает Pydantic-результат по схеме сценария и формирует краткий отчёт
менеджеру (human-in-the-loop). Пишет JSON и лог транскрипта в results/.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .providers import LLMProvider
from .state import OUTCOMES, ConversationState, LprResult, QualifyResult

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# Почта, продиктованная голосом, часто приходит мусором от ASR («useraol.com»,
# без @). Сохраняем только синтаксически валидный адрес, иначе — пусто.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def _valid_email(s: str | None) -> str:
    m = _EMAIL_RE.search((s or "").replace(" ", ""))
    return m.group(0) if m else ""

_EXTRACT_PROMPT_A = (
    "Ты извлекаешь структуру из транскрипта холодного звонка ЧЗЭ (квалификация по "
    "крючку «истёкший срок службы подстанций / ЭПБ»). Верни СТРОГО JSON с полями:\n"
    "  company, phone,\n"
    "  outcome — одно из: connected_lpr | data_extracted | callback_scheduled | "
    "not_qualified | refused,\n"
    "  qualified (bool),\n"
    "  binary_filter — underground_yes | underground_no | unknown,\n"
    "  lpr_name, lpr_role, lpr_email, lpr_phone_ext,\n"
    "  park: {total (int), types (список из КТПВШ/КТПВМ/КТПРН), expired_or_on_epb "
    "(int — сколько за сроком/на продлении ЭПБ), last_replacement_year (int|null)},\n"
    "  purchase: {plan (строка), channel — тендер|прямой договор|неизвестно},\n"
    "  decision_chain (список ролей),\n"
    "  callback_at (YYYY-MM-DD|null),\n"
    "  secretary_name, notes (КРАТКО, до 200 символов).\n"
    "Бери только то, что реально прозвучало. Чего нет — пустая строка / 0 / null / "
    "[]. outcome обязан быть непустым и осмысленным. Верни только JSON, без пояснений."
)
_EXTRACT_PROMPT_B = (
    "Ты извлекаешь структуру из транскрипта звонка по поиску ЛПР. Верни СТРОГО JSON "
    "с полями: company, phone, target_role, lpr_name, lpr_mobile, method_used "
    "('1'/'2'/'3' — какой метод сработал), notes. Если данных нет — пустая строка."
)


def _transcript_text(state: ConversationState) -> str:
    lines = []
    for t in state.transcript:
        if t.role in ("agent", "interlocutor"):
            who = "АГЕНТ" if t.role == "agent" else "СОБЕСЕДНИК"
            lines.append(f"{who}: {t.text}")
    return "\n".join(lines)


def extract(state: ConversationState, llm: LLMProvider):
    """Извлечь результат сценария из транскрипта."""
    prompt = _EXTRACT_PROMPT_A if state.mode == "A" else _EXTRACT_PROMPT_B
    if state.mode == "A":
        today = datetime.now(timezone.utc).date().isoformat()
        prompt += (f"\nСегодняшняя дата: {today}. Вычисляй callback_at из "
                   f"относительных фраз («через две недели», «в октябре») от неё.")
    raw = llm.complete(
        system=prompt,
        messages=[{"role": "user", "content": _transcript_text(state)}],
        model=llm.postprocess_model,
        max_tokens=1200,   # схема A объёмная — запас, чтобы JSON не усекался
        temperature=0,
    )
    data = _parse_json(raw)
    # Компанию и телефон знаем достоверно из входа — перезаписываем (а не setdefault,
    # иначе LLM может подставить «нашу» компанию или оставить пустой телефон).
    if state.company:
        data["company"] = state.company
    data["phone"] = state.phone
    if state.mode == "A":
        # Поля, которые надёжнее взять из state (мы их детерминированно отслеживали).
        data["uks_triggered"] = state.uks_triggered
        if state.secretary_name and not data.get("secretary_name"):
            data["secretary_name"] = state.secretary_name
        if state.binary_filter != "unknown" and data.get("binary_filter") in (None, "", "unknown"):
            data["binary_filter"] = state.binary_filter
        data["outcome"] = _normalize_outcome(data.get("outcome"), state)
        if data.get("lpr_email"):
            data["lpr_email"] = _valid_email(data["lpr_email"])
        return QualifyResult(**_filter(data, QualifyResult))
    data.setdefault("target_role", state.target_role or "")
    return LprResult(**_filter(data, LprResult))


def _normalize_outcome(value, state: ConversationState) -> str:
    """Гарантировать непустой осмысленный outcome (Правило №1, скрипт §6)."""
    if value in OUTCOMES:
        return value
    if state.refused:
        return "refused"
    if state.binary_filter == "underground_no":
        return "not_qualified"
    if state.reached_lpr:
        return "connected_lpr"
    # Есть ли хоть какие-то извлечённые данные в транскрипте?
    text = " ".join(t.text.lower() for t in state.transcript if t.role == "interlocutor")
    if "перезвон" in text or "наберите" in text:
        return "callback_scheduled"
    if "@" in text or "добавочн" in text or "зовут" in text or "отчеств" in text:
        return "data_extracted"
    return "not_qualified"


def _filter(data: dict, model) -> dict:
    return {k: v for k, v in data.items() if k in model.model_fields}


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw[raw.find("{"):]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return {}


def report(result) -> str:
    """Краткий отчёт менеджеру (human-in-the-loop) с выделением «горячих» признаков."""
    d = result.model_dump()
    lines = ["=== ОТЧЁТ ПО ЗВОНКУ (для менеджера) ==="]

    # «Горячие» признаки сверху (только для QualifyResult сценария A).
    if isinstance(result, QualifyResult):
        hot: list[str] = []
        if result.park.expired_or_on_epb > 0:
            hot.append(f"🔥 {result.park.expired_or_on_epb} подстанций за сроком / на ЭПБ")
        nxt = []
        if result.lpr_email:
            nxt.append(f"почта ЛПР {result.lpr_email}")
        if result.callback_at:
            nxt.append(f"перезвон {result.callback_at}")
        if result.outcome == "connected_lpr":
            nxt.append("дошли до ЛПР")
        if nxt:
            hot.append("➡ следующий шаг: " + ", ".join(nxt))
        if hot:
            lines.append("  " + "  |  ".join(hot))
            lines.append("  " + "-" * 40)

    for k, v in d.items():
        lines.append(f"  {k}: {v if v not in ('', [], None) else '—'}")
    lines.append("Решение о передаче лида в работу принимает менеджер.")
    return "\n".join(lines)


def save(state: ConversationState, result) -> Path:
    """Сохранить JSON-результат и лог транскрипта в results/."""
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = RESULTS_DIR / f"{ts}_mode{state.mode}"
    base.with_suffix(".json").write_text(
        json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    base.with_name(base.name + "_transcript").with_suffix(".log").write_text(
        _transcript_text(state), encoding="utf-8")
    return base.with_suffix(".json")
