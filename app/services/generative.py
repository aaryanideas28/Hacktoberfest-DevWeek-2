"""
LLM provider chain (app/services/generative.py).

    rewrite(prompt, rate)  -> shorter prompt          (used by compressor.py, behavior unchanged)
    complete(instr, ...)   -> raw model text          (used by gemma.py for handoff extraction)

Both go through the same chain: Google AI Studio first, OpenRouter second, with a response
cache and a circuit breaker so a dead provider is skipped for 45 seconds.
"""

import hashlib
import os
import time

import requests

_CACHE = {}
_CACHE_MAX = 256
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


def _google(instr, max_tokens=2048, temperature=0.1, json_mode=False, timeout=TIMEOUT):
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("no google key")
    model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
    cfg = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if json_mode:
        cfg["responseMimeType"] = "application/json"
    if "2.5" in model:
        cfg["thinkingConfig"] = {"thinkingBudget": 0}   # no "thinking" = much faster
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"contents": [{"parts": [{"text": instr}]}], "generationConfig": cfg},
        timeout=timeout,
    )
    r.raise_for_status()
    return _clean(r.json()["candidates"][0]["content"]["parts"][0]["text"])


def _openrouter(instr, max_tokens=2048, temperature=0.1, json_mode=False, timeout=TIMEOUT):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("no openrouter key")
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
              "temperature": temperature, "max_tokens": max_tokens,
              "messages": [{"role": "user", "content": instr}]},
        timeout=timeout,
    )
    r.raise_for_status()
    return _clean(r.json()["choices"][0]["message"]["content"])


def _remember(ck, out):
    if len(_CACHE) >= _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)))      # drop the oldest entry
    _CACHE[ck] = out


def _run_chain(instr, ck, use_cache=True, **kw):
    if use_cache and ck in _CACHE:
        return _CACHE[ck]
    last = None
    for name, fn in (("google", _google), ("openrouter", _openrouter)):
        if _DOWN.get(name, 0) > time.time():
            continue                      # circuit breaker: provider failed recently
        t = time.perf_counter()
        try:
            out = fn(instr, **kw)
            print(f"[generative] {name} ok {int((time.perf_counter()-t)*1000)} ms")
            if out:
                if use_cache:
                    _remember(ck, out)
                return out
        except Exception as e:
            _DOWN[name] = time.time() + 45
            last = e
            print(f"[generative] {name} failed ({e}); skipping it for 45s")
    raise RuntimeError(f"all providers failed: {last}")


def rewrite(prompt: str, rate: float) -> str:
    ck = hashlib.sha256(f"rewrite|{round(rate, 2)}|{prompt}".encode()).hexdigest()
    return _run_chain(_prompt(prompt, rate), ck)


def complete(instr: str, max_tokens: int = 2048, temperature: float = 0.1,
             json_mode: bool = False, timeout: int = TIMEOUT, use_cache: bool = True) -> str:
    """Raw completion through the same provider chain (used for handoff extraction)."""
    ck = hashlib.sha256(f"complete|{max_tokens}|{temperature}|{json_mode}|{instr}".encode()).hexdigest()
    return _run_chain(instr, ck, use_cache=use_cache, max_tokens=max_tokens,
                      temperature=temperature, json_mode=json_mode, timeout=timeout)