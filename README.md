# ⚡ ContextSwitch

> **Open-Source Neural Engine for Prompt Compression & Context Handoff**  
> Powered by **Google Gemma 2 2B (Token-Level Perplexity Pruning)** & instrumented with **Sentry Tracing & Metrics**.

---

## 🎯 What is ContextSwitch?
Developers constantly hit context limits, burn tokens, and lose technical constraints when migrating conversations between AI models (ChatGPT, Claude, Gemini, Cursor). 

**ContextSwitch** solves this with two core capabilities:
1. **Gemma 2 2B Information-Theoretic Prompt Compression:** Performs a single forward pass through Gemma 2 2B to compute token-level logits and surprisal ($I(x_i) = -\log P(x_i \mid x_{<i})$). It drops the lowest-entropy predictable tokens without sequential decoding (`model.generate()`), achieving sub-millisecond execution with **zero output token costs**.
2. **Lossless Context Handoff:** Extracts state (decisions, anti-patterns, constraints, next tasks) into a clean, portable System Prompt markdown artifact for any target LLM.
3. **Sentry Observability:** Custom spans trace token reduction, latency, and compression ratios in real time.

---

## 🔬 How Gemma 2 2B Token-Level Compression Works

```
 ┌──────────┐      ┌───────────────────────────┐
 │ User UI  ├─────►│ FastAPI / Orchestration   │
 └──────────┘      └─────────────┬─────────────┘
                                 │
                                 ▼
                   ┌───────────────────────────┐
                   │  Gemma 2 2B Local Engine  │◀─── [Calculates token-by-token]
                   │    (PyTorch / HF / vLLM)  │     [logits & perplexity mask]
                   └─────────────┬─────────────┘
                                 │ (Drops lowest-entropy tokens, e.g. bottom 40%)
                                 ▼
 ┌──────────┐      ┌───────────────────────────┐
 │Target LLM│◄─────┤ Raw Token Slice / String  │
 └──────────┘      └───────────────────────────┘
```

- **Single Forward Pass:** Evaluates token surprisal in parallel; avoids slow autoregressive token generation loops.
- **Zero Generation Cost:** Does not consume output tokens on the compression model.
- **Syntactic Protection:** Preserves fenced code blocks (` ```...``` `), identifiers, and boundary tokens.
- **Adaptive Pruning Intensity:** Configurable `drop_rate` (20% conservative to 50% aggressive).

---

## 🚀 Quickstart

### 1. Clone & Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Credentials (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in:
- `GEMINI_API_KEY`: Google AI Studio API key (optional for cloud Gemma 2 fallback).
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
All 22 unit & integration tests pass across compression, extraction, and API endpoints.
