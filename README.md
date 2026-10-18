# 📦 ContextPack

**Compress your prompts. Carry your context.**

ContextPack is a FastAPI service with a small web UI that does two jobs:

1. **Prompt compression**: shortens a prompt while keeping every instruction, number, negation and quoted string intact.
2. **Context handoff**: turns a long chat transcript into a compact, structured artifact you can paste into a new chat, so a new assistant can pick up where the last one stopped.

Both features share one safety principle: **nothing a model says is trusted**. Every model output is checked against the original text before it is returned, and the service falls back to deterministic rules when a check fails.

## 🗺️ System Architecture

```text
               ┌──────────────────────────────────────┐
               │         Web UI (app/static)          │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │          FastAPI (main.py)           │
               └──────┬────────────────────────┬──────┘
                      │                        │
        POST /api/v1/compress            POST /api/v1/handoff
                      │                        │
                      ▼                        ▼
         ┌────────────────────────┐  ┌────────────────────────┐
         │      compressor.py     │  │      extractor.py      │
         │  (Rule-based pre-pass) │  │  (Rules, always runs)  │
         └────────────┬───────────┘  └────────────┬───────────┘
                      │                           │
                      ▼                           ▼
         ┌────────────────────────┐  ┌────────────────────────┐
         │      LLM Rewrite       │  │        gemma.py        │
         │   (Generative pass)    │  │(Extraction, optional)  │
         └────────────┬───────────┘  └────────────┬───────────┘
                      │                           │
                      ▼                           ▼
         ┌────────────────────────┐  ┌────────────────────────┐
         │  validation (protect.py)│  │ Verify vs Transcript   │
         └────────────┬───────────┘  └────────────┬───────────┘
                      │                           │
                      ▼                           ▼
         ┌────────────────────────┐  ┌────────────────────────┐
         │     Fallback Layer     │  │ Merge & Create Artifact│
         └────────────┬───────────┘  └────────────┬───────────┘
                      │                           │
                      └─────────────┬─────────────┘
                                    │
                                    ▼
               ┌──────────────────────────────────────┐
               │    LLM Engine Orchestrator           │
               │    (generative.py / Google AI Studio)│
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │       OpenRouter Integration         │
               │    (Cache + 45s Circuit Breaker)     │
               └──────────────────┬───────────────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────────┐
               │  Optional Local Compression Engine   │
               │  (gemma_entropy.py / Gemma 2 2B)     │
               │  - Mathematical Surprisal Pruning    │
               └──────────────────────────────────────┘
```


### Project layout
app/
├── core/               Sentry tracing
├── routers/
│   ├── compress.py     /api/v1/compress
│   └── handoff.py      /api/v1/handoff and snapshot endpoints
├── services/
│   ├── compressor.py   compression pipeline and validation
│   ├── protect.py      protected-token detection
│   ├── generative.py   LLM provider chain (rewrite + complete)
│   ├── gemma.py        handoff extraction via the provider chain
│   ├── extractor.py    rule-based extraction, verification, Markdown formatter
│   └── gemma_entropy.py  optional local Gemma 2 2B engine
├── static/             index.html, app.js, style.css
└── db.py               SQLite storage for snapshots
tests/                  test_api, test_compressor, test_extractor, test_jumbo, test_main
main.py                 app entry point
---

## ✂️ Prompt compression

### Modes

| `mode` | What it does |
|---|---|
| `hybrid` (default) | Rule pass, then an LLM rewrite, then validation. Unknown modes behave like `hybrid`. |
| `generative` | Same pipeline as `hybrid` (kept for the UI option). |
| `rule` | Deterministic regex cleanup only: filler phrases, wordy phrases, greetings. No network. |
| `gemma_entropy` (alias `entropy`) | Local Gemma 2 2B surprisal pruning. Optional; see below. |

`drop_rate` (default `0.40`) is the target reduction. The LLM rewrite is asked to be about that much shorter, and the entropy engine drops that fraction of non-protected words.

### How it stays safe

- **Protected spans are never touched:** negations (`not`, `never`, `don't` ...), numbers, quoted strings, ALL-CAPS words, `Label:` patterns, `[placeholders]`, URLs, e-mail addresses, non-ASCII text, fenced and inline code, and anything inside `<data>...</data>`.
- **Every LLM rewrite is validated** before it is accepted: it must be shorter, must not start with chat preamble, must reuse the original vocabulary (rewritten, not answered), must keep every protected span verbatim and case-sensitively, and must pass a meaning audit (no dropped instruction words such as `only`, `exactly`, `tone`, `format`; limited new or lost content words).
- **Final check:** the output is re-checked with `missing_critical()`.
- **Fallbacks:** a rejected rewrite falls back to the rule-only output; the original prompt is returned unchanged only as a last resort (`engine = "passthrough"`).
- **Speed guards:** prompts up to 1,200 tokens are rewritten in one call. Larger prompts send only their biggest prose paragraphs (at most 3, in parallel) and never send code fences or `<data>` blocks. The whole LLM stage has an 8 second deadline.

**Tip:** wrap customer emails, logs or other payloads in `<data>...</data>` so only your instructions are compressed.

### Example

```bash
curl -X POST http://localhost:8000/api/v1/compress \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Can you please help me write unit tests? Thank you so much!", "mode": "hybrid"}'
```

Response fields: `original_prompt`, `compressed_prompt`, `mode_used`, `engine_used`, `original_tokens`, `compressed_tokens`, `tokens_saved`, `compression_ratio`, `latency_ms`.

### Optional: local Gemma 2 2B entropy engine

`gemma_entropy` runs one forward pass through Gemma 2 2B, averages token surprisal per word, and drops the most predictable unprotected words. It makes no generation calls, but it is **not instant**: expect tens to hundreds of milliseconds on a GPU and much longer on CPU.

```bash
pip install torch transformers
export HF_TOKEN=your_huggingface_token   # Gemma is gated: accept the license on Hugging Face first
```

If the model cannot be loaded, the engine falls back to a safe filler-word filter (it only deletes words like `a`, `the`, `very`) and reports `protected-filler-fallback` as the engine. Set `GEMMA_ENTROPY_WINDOW` to change the tokens-per-pass window (default 2048).

---

## 🔄 Context handoff

`POST /api/v1/handoff` takes `project_name` and `raw_transcript` and returns a Markdown artifact to paste into a new chat.

How it works:

1. **Rule-based extraction always runs.** It detects tech stack, decisions, completed work, constraints, failed approaches, open issues, key files and endpoints, and the most recent next step. Nothing is invented: empty sections stay empty.
2. **LLM extraction runs if a provider key is set.** Long transcripts are split into chunks (30,000 characters each, up to 6, extracted in parallel) and merged.
3. **The LLM output is verified, never trusted.** Items whose numbers or words cannot be found in the transcript are removed, constraints the model missed or softened are restored from the rule-based pass, and file names and tech stack found by regex are merged in.
4. The artifact is formatted, token savings are computed, and a snapshot is saved to SQLite.

The artifact contains: project summary, tech stack, decisions, completed work, strict constraints, failed approaches, open issues, key files and endpoints, and the immediate next task.

```bash
curl -X POST http://localhost:8000/api/v1/handoff \
  -H "Content-Type: application/json" \
  -d '{"project_name": "MyProject", "raw_transcript": "We decided to use SQLite. You must not change the API format. Next, add auth."}'
```

Response fields: `snapshot_id`, `project_name`, `structured_data`, `formatted_markdown`, `tokens_saved`, `original_tokens`, `artifact_tokens`, `engine` (`gemma-api (verified against transcript)` or `rule-based`), and `notes` (what was removed or added during verification).

### Switching chats: snapshot endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/handoff/latest?project_name=X` | Most recent artifact for a project, the one to start a new chat from |
| `GET /api/v1/handoff/{snapshot_id}` | A specific snapshot |
| `GET /api/v1/handoff?project_name=X&limit=20` | List snapshots, newest first |
| `GET /api/v1/projects` | Projects with snapshot counts |

---

## 🚀 Quickstart

### 1. Clone and set up

```bash
git clone https://github.com/aaryanideas28/Hacktoberfest-DevWeek-2.git
cd Hacktoberfest-DevWeek-2
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env     # Windows: copy .env.example .env
```

Set at least one LLM key. Without any key the app still works, using rule-based compression and extraction.

### 3. Run

```bash
uvicorn main:app --reload
```

Open <http://localhost:8000> for the UI, or <http://localhost:8000/docs> for the interactive API docs.

### 4. Test

```bash
pytest
```

---

## ⚙️ Configuration

| Variable | Purpose | Default |
|---|---|---|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Google AI Studio key (first provider) | none |
| `GOOGLE_MODEL` | Google model name | `gemini-2.5-flash` |
| `OPENROUTER_API_KEY` | OpenRouter key (second provider) | none |
| `OPENROUTER_MODEL` | OpenRouter model name | `google/gemma-4-31b-it:free` |
| `HF_TOKEN` | Hugging Face token for the local Gemma 2 2B engine | none |
| `GEMMA_ENTROPY_WINDOW` | Tokens per forward pass for the local engine | `2048` |
| `CONTEXTSWITCH_DB` | Path to the SQLite file | `contextswitch.db` in the project root |
| `SENTRY_DSN` | Enables Sentry tracing | none |

Provider behavior: Google is tried first, then OpenRouter. A provider that fails is skipped for 45 seconds, and identical requests are served from an in-memory cache (256 entries).

`GET /api/v1/health` reports whether an LLM provider and Sentry are configured.

---

## ⚠️ Limitations

- Compression and extraction rely on heuristics plus an LLM. Review important prompts and handoffs before relying on them.
- Verification is deliberately strict. An extracted item whose number is written differently from the transcript (for example "three" instead of "3") is removed, and the `notes` field says so.
- For short transcripts the handoff artifact can be longer than the transcript, because of its preamble and section headers. Savings appear on longer conversations.
- Token counts use the Gemma tokenizer when it is loaded, otherwise an estimate of about 4 characters per token.