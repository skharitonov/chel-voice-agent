# CLAUDE.md

Инструкции для Claude Code при работе в этом репозитории.

## Что это

Прототип голосового AI-агента (AI SDR) для холодного обзвона ЧЗЭ — производителя
взрывозащищённых трансформаторных подстанций. Два сценария: **A** (квалификация
лида), **B** (поиск ЛПР по методу Соколова). Эксперимент, не production.

Перед изменениями прочитай:
- [`docs/specs/2026-06-10-voice-agent.md`](docs/specs/2026-06-10-voice-agent.md) — PRD (что и зачем)
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — как устроено
- [`TASK.md`](TASK.md) — исходная постановка

## Стек

- **Python 3.11+**, оркестрация на **LiveKit Agents SDK**
- Архитектура голоса: **STT → LLM → TTS** (НЕ end-to-end — см. ARCHITECTURE §1)
- STT: Deepgram Nova-2 / Whisper · LLM: Claude Sonnet 4.6 (реплики) + Haiku 4.5
  (постобработка) · TTS: Inworld TTS 1.5 Mini
- Структурированный вывод: **Pydantic**

## Структура

```
voice_agent/      пакет ядра
  state.py        ConversationState + Pydantic-результаты (QualifyResult, LprResult)
  providers.py    интерфейсы STTProvider/LLMProvider/TTSProvider + реализации
  director.py     DialogueDirector — стадии и тактики (эскалация M1→M3)
  objections.py   ObjectionHandler — ответы на возражения блокера
  compliance.py   ComplianceGuard — проверка реплик до отправки
  extractor.py    JSONExtractor + Reporter — постобработка транскрипта
  text_sim.py     текстовая симуляция (фаза 1)
  voice_core.py   LiveKit голосовой цикл (фаза 2) — console-режим
  lk_llm.py       адаптер: наш конвейер как LiveKit llm.LLM (ChelLLM)
prompts/          системные промпты как md-файлы (+ shared/)
scripts/          qualify.py (mode A), find_lpr.py (mode B) — тонкие entrypoints
results/          JSON + транскрипты (gitignored)
```

## Команды

```bash
pip install -r requirements.txt
python scripts/qualify.py --phone "+7..." --company "..." --mode A   # сценарий A (текст)
python scripts/find_lpr.py --phone "+7..." --role "главный энергетик" --mode B  # сценарий B
```

## Конвенции

- **Промпты — это md-файлы** в `prompts/`, не строковые литералы в коде. Логика
  диалога правится в промптах, код их загружает.
- **Провайдеры за интерфейсами** (`providers.py`). Смена STT/LLM/TTS не должна
  затрагивать director/objections/compliance/extractor.
- **Промпт — это набор тактик, а не жёсткий скрипт** (вывод ресёрча: «режиссура
  диалога»). Director выбирает тактику, LLM импровизирует в её рамках.
- **Фазность:** сначала текстовая симуляция (фаза 1), потом голос (фаза 2). Один
  и тот же конвейер; различается только STT/TTS-слой.
- Тексты, промпты, отчёты — **на русском**.

## Безопасность и комплаенс

- **Не коммить** `.env` и `results/` (в `.gitignore`). Секреты — только в `.env`.
- Сценарий B — авторизованный внутренний инструмент. Принцип: **эвазия, а не
  прямая ложь** (см. PRD §6). `ComplianceGuard` обязателен в конвейере — не
  обходить его при рефакторинге.

## Чего не делать

- Не превращать прототип в полный пайплайн продаж/CRM (узкий scope: 1 голос + супервайзеры).
- Не подключать реальную телефонию на +7 без отдельной задачи (фаза 3, вне scope).
- Не заменять STT→LLM→TTS на end-to-end модель без пересмотра PRD/ARCHITECTURE.
