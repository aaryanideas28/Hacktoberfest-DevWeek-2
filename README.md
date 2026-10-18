# ⚡ ContextSwitch

> **Open-Source Neural Engine for Prompt Compression & Context Handoff**  
> Powered by **Google Gemma 2** & instrumented with **Sentry Tracing & Metrics**.

---

## 🎯 What is ContextSwitch?
Developers hit context limits, burn tokens, and lose technical constraints when migrating conversations between AI models (ChatGPT, Claude, Gemini, Cursor). 

**ContextSwitch** solves this with two core capabilities:
1. **Aggressive Prompt Compression:** Blends deterministic regex abstractions with Google's Gemma 2 to strip fluff and preserve code blocks.
2. **Lossless Context Handoff:** Compiles messy developer transcripts into portable, structured System Prompt artifacts ready for your next AI session.
3. **Sentry Observability:** Custom spans trace token reduction, latency, and compression ratios in real time.

---

## 📂 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 ContextSwitch Studio UI                 │
│               (FastAPI + Web UI Frontend)               │
└───────────┬─────────────────────────────────┬───────────┘
            │                                 │
            ▼                                 ▼
┌─────────────────────────┐       ┌─────────────────────────┐
│    Prompt Compressor    │       │     Context Handoff     │
│ (Rule-based + Gemma 2)  │       │ (Structured Extraction) │
└───────────┬─────────────┘       └───────────┬─────────────┘
            │                                 │
            ▼                                 ▼
┌─────────────────────────────────────────────────────────┐
│              Sentry Agent Tracing & Spans               │
│        (Monitors token savings, latency & errors)       │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart

### 1. Clone & Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in:
- `GEMINI_API_KEY`: Get a free key at [Google AI Studio](https://aistudio.google.com) (enables Gemma 2).
- `SENTRY_DSN`: Your Sentry project DSN for tracing.

### 3. Launch ContextSwitch Studio
```bash
uvicorn main:app --reload
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🧪 Running Tests
```bash
pytest -v
```
