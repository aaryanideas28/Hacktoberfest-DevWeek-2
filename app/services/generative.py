import os, time, hashlib, requests

_CACHE = {}
_DOWN = {}      # provider -> timestamp until which it is skipped
TIMEOUT = 12


def _prompt(text, rate):
    return (
        f"Rewrite the prompt below to be about {int(rate*100)}% shorter while keeping the exact meaning. "
        "Copy every number, quoted string, URL, ALL-CAPS word, negation (NOT, not, never, don't), "
        "code span and anything inside <data>...</data> EXACTLY as written, character for character. "
        "Never change letter case. Do NOT answer the prompt. "
        "Prefer deleting words over rephrasing. Never remove instructions about format, tone, level of detail, audience, language or length."
        "Output ONLY the rewritten prompt, no preface, no quotes, no markdown.\n\nPROMPT:\n" + text
    )


def _clean(out):
    out = (out or "").strip()
    if out.startswith("```"):
        out = out.strip("`").strip()
    return out


def _google(instr):
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("no google key")
    model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    cfg = {"temperature": 0.1, "maxOutputTokens": 2048}
    if "2.5" in model:
        cfg["thinkingConfig"] = {"thinkingBudget": 0}   # no "thinking" = much faster
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"contents": [{"parts": [{"text": instr}]}], "generationConfig": cfg},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return _clean(r.json()["candidates"][0]["content"]["parts"][0]["text"])


def _openrouter(instr):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("no openrouter key")
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": "google/gemma-4-31b-it:free", "temperature": 0.1,
              "messages": [{"role": "user", "content": instr}]},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return _clean(r.json()["choices"][0]["message"]["content"])


def rewrite(prompt: str, rate: float) -> str:
    ck = hashlib.sha256(f"{round(rate, 2)}|{prompt}".encode()).hexdigest()
    if ck in _CACHE:
        return _CACHE[ck]
    instr = _prompt(prompt, rate)
    last = None
    for name, fn in (("google", _google), ("openrouter", _openrouter)):
        if _DOWN.get(name, 0) > time.time():
            continue                      # circuit breaker: provider failed recently
        t = time.perf_counter()
        try:
            out = fn(instr)
            print(f"[generative] {name} ok {int((time.perf_counter()-t)*1000)} ms")
            if out:
                _CACHE[ck] = out
                return out
        except Exception as e:
            _DOWN[name] = time.time() + 45
            last = e
            print(f"[generative] {name} failed ({e}); skipping it for 45s")
    raise RuntimeError(f"all providers failed: {last}")