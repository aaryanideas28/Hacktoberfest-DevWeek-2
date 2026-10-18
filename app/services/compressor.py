"""
Unified prompt compression pipeline (app/services/compressor.py).

Modes:
    "rule"          regex filler-phrase stripper (protected spans are masked first)
    "hybrid"        rule pass, then Gemma/Gemini API rewrite, then validation (default)
    "generative"    same pipeline as hybrid (kept for the UI option)
    "gemma_entropy" local Gemma 2 2B surprisal pruning (alias: "entropy")
    anything else   treated as "hybrid"

Safety design (nothing the model says is trusted):
    1. Every API rewrite must pass _acceptable(): a real, shorter string that keeps every
       protected span (negations, numbers, quotes, ALL-CAPS labels, <data>, code, URLs)
       VERBATIM and CASE-SENSITIVE, with no preamble / answering / markdown fences.
    2. The final output is checked again with protect.missing_critical().
    3. If anything fails, we fall back to the rule-only output, and only as the very last
       resort to the original prompt (engine = "passthrough").
    4. One attempt per request for API modes (a lower drop rate means nothing to a
       generative rewrite and each retry costs seconds).
"""

import re
import time
from typing import Any, Dict, Optional

from app.services.gemma_entropy import compress_via_gemma_entropy, count_tokens
from app.services.generative import rewrite
from app.services.protect import mask, missing_critical, protected_spans, tidy, unmask

# Longer phrases FIRST, so "thank you so much" is removed whole and not as "thank you" + "so much".
FILLER_PATTERNS = [
    r"\bas an ai language model\b,?",
    r"\bthank you (?:so|very) much\b[,!. ]*",
    r"\bthanks (?:so|very) much\b[,!. ]*",
    r"\bthanks a lot\b[,!. ]*",
    r"\bi (?:really )?appreciate (?:your|the) help\b[,!. ]*",
    r"\bi am wondering if you could\b",
    r"\bi would like you to\b",
    r"\bi want you to\b",
    r"\bcan you please\b",
    r"\bcould you kindly\b",
    r"\bcould you please\b",
    r"\bcould you\b",
    r"\bhelp me to\b",
    r"\bplease\b,?",
    r"\bkindly\b,?",
    r"\bthank you\b[,!. ]*",
    r"\bthanks\b[,!. ]*",
]
_GREETING = re.compile(r"^[ \t]*(?:hello|hi|hey)\b[ \t]*[!,.]*[ \t]*", re.I | re.M)
_PREAMBLES = ("here is", "here's", "sure", "certainly", "of course", "rewritten prompt",
              "shortened prompt", "compressed prompt", "okay,", "ok,")


def estimate_tokens(text: str) -> int:
    """Heuristic estimation (~4 characters per token). Kept for backward compatibility."""
    return max(1, len(text.strip()) // 4) if text.strip() else 0


# --------------------------------------------------------------------------- #
# Rule-based stage
# --------------------------------------------------------------------------- #
def _clean_punct(s: str) -> str:
    """Repair punctuation left behind after phrases are deleted. Idempotent and path/.env safe."""
    s = re.sub(r"[ \t]*,[ \t]*([.!?])", r"\1", s)          # ",."  ", !"  -> "." / "!"
    s = re.sub(r"\.[ \t]*!", ".", s)                         # ".!" -> "."
    s = re.sub(r"(?<=[^\s.])\.\.(?=\s|$)", ".", s)           # "files.." -> "files." (keeps "../x")
    s = re.sub(r"[ \t]+([.!?,;:])(?=\s|$)", r"\1", s)        # "word ." -> "word." (keeps ".env")
    s = re.sub(r"(?m)^[ \t]*[,;:]+[ \t]*", "", s)            # orphan comma at line start
    s = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", s)            # double spaces inside a line
    s = re.sub(r",\s*$", "", s)                              # dangling trailing comma
    return s.strip()


def _rule_strip(prompt: str) -> str:
    """Strip filler phrases while protected spans (code, quotes, numbers, <data>...) are masked."""
    masked, store = mask(prompt)
    for pattern in FILLER_PATTERNS:
        masked = re.sub(pattern + r"[ \t]*", "", masked, flags=re.IGNORECASE)
    masked = _GREETING.sub("", masked)
    masked = tidy(masked)
    masked = _clean_punct(masked)
    return unmask(masked, store).strip()


def compress_prompt_rule_based(prompt: str) -> Dict[str, Any]:
    """Deterministic regex compression that never touches protected spans."""
    start = time.perf_counter()
    if not prompt or not prompt.strip():
        return {
            "compressed": "", "compressed_prompt": "", "original_tokens": 0, "compressed_tokens": 0,
            "tokens_saved": 0, "compression_ratio": 0.0, "latency_ms": 0.0, "engine": "rule-based",
        }
    compressed = _rule_strip(prompt)
    orig_t = count_tokens(prompt)
    comp_t = count_tokens(compressed)
    saved = max(0, orig_t - comp_t)
    return {
        "compressed": compressed,
        "compressed_prompt": compressed,
        "original_tokens": orig_t,
        "compressed_tokens": comp_t,
        "tokens_saved": saved,
        "compression_ratio": round((saved / orig_t) * 100, 2) if orig_t > 0 else 0.0,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "engine": "rule-based",
    }


# --------------------------------------------------------------------------- #
# Validation of untrusted model output
# --------------------------------------------------------------------------- #
def _unwrap(cand: str, source: str) -> str:
    """Remove a ```markdown fence``` the model added around its answer."""
    cand = cand.strip()
    if cand.startswith("```") and cand.endswith("```") and "```" not in source:
        cand = re.sub(r"^```[\w-]*\n?", "", cand)
        cand = re.sub(r"\n?```$", "", cand)
    return cand.strip()


def _words(s: str):
    return set(re.findall(r"\w+", s.lower()))


def _acceptable(source: str, cand) -> bool:
    """
    True only if `cand` is a trustworthy rewrite of `source`:
    a non-empty str, strictly fewer tokens, no chatty preamble, mostly the same vocabulary
    (i.e. rewritten, not answered), and every protected span of the source appears
    verbatim (case-sensitive) in it.
    """
    if not isinstance(cand, str) or not cand.strip():
        return False
    cand = _unwrap(cand, source)
    if not cand:
        return False
    if count_tokens(cand) >= count_tokens(source):
        return False
    low = cand.lower().lstrip()
    src_low = source.lower().lstrip()
    if low.startswith(_PREAMBLES) and not src_low.startswith(_PREAMBLES):
        return False
    cw, sw = _words(cand), _words(source)
    if cw and len(cw & sw) / len(cw) < 0.5:
        return False
    return all(source[s:e] in cand for s, e in protected_spans(source))


def _missing(original: str, compressed: str) -> Dict[str, int]:
    """missing_critical() that can never raise; any error counts as 'something missing'."""
    try:
        return dict(missing_critical(original, compressed) or {})
    except Exception:  # noqa: BLE001
        return {"<check error>": 1}


# --------------------------------------------------------------------------- #
# Single attempt + final packaging
# --------------------------------------------------------------------------- #
def _api_rewrite_attempt(prompt: str, rate: float, label: str) -> Dict[str, str]:
    """Rule pass, then validated API rewrite. Always returns usable text."""
    ruled = _rule_strip(prompt)
    try:
        cand = rewrite(ruled, rate)
        if _acceptable(ruled, cand):
            return {"compressed_prompt": _unwrap(cand, ruled), "engine": label}
        print("[Compressor] rewrite rejected (not shorter / protected text changed / not a rewrite)")
    except Exception as e:  # noqa: BLE001
        print(f"[Compressor] API failed ({e}); using rule fallback")
    return {"compressed_prompt": ruled, "engine": "rule-based (API fallback)"}


def _compress_once(prompt: str, mode: str, rate: float) -> Dict[str, str]:
    """One compression attempt. Returns {"compressed_prompt", "engine"}."""
    if mode in ("gemma_entropy", "entropy"):
        res = compress_via_gemma_entropy(prompt, drop_rate=rate)
        return {"compressed_prompt": res["compressed_prompt"], "engine": res["engine"]}

    if mode == "rule":
        return {"compressed_prompt": _rule_strip(prompt), "engine": "rule-based"}

    if mode == "generative":
        return _api_rewrite_attempt(prompt, rate, "gemma-api-rewrite")

    # hybrid (default) and any unknown mode
    return _api_rewrite_attempt(prompt, rate, "hybrid (rule + gemma-api-rewrite)")


def _finalize(
    original: str,
    compressed: str,
    engine: str,
    rate_used: float,
    start: float,
    missing: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    orig_t = count_tokens(original)
    comp_t = count_tokens(compressed)
    saved = max(0, orig_t - comp_t)
    return {
        "compressed": compressed,
        "compressed_prompt": compressed,
        "original_tokens": orig_t,
        "compressed_tokens": comp_t,
        "tokens_saved": saved,
        "compression_ratio": round((saved / orig_t) * 100, 2) if orig_t > 0 else 0.0,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "engine": engine,
        "drop_rate_used": rate_used,
        "critical_missing": missing or {},
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "compressed": "", "compressed_prompt": "", "original_tokens": 0, "compressed_tokens": 0,
        "tokens_saved": 0, "compression_ratio": 0.0, "latency_ms": 0.0, "engine": "none",
        "drop_rate_used": 0.0, "critical_missing": {},
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def compress_prompt(
    prompt: str,
    mode: str = "hybrid",
    drop_rate: float = 0.40,
) -> Dict[str, Any]:
    """
    Unified prompt compression with layered safety.

    Returns dict with: compressed, compressed_prompt, original_tokens, compressed_tokens,
    tokens_saved, compression_ratio, latency_ms, engine, drop_rate_used, critical_missing.
    """
    start = time.perf_counter()
    mode_lower = (mode or "hybrid").strip().lower() if isinstance(mode, str) else "hybrid"

    if not isinstance(prompt, str) or not prompt.strip():
        return _empty_result()

    try:
        rate = float(drop_rate if drop_rate is not None else 0.40)
    except (TypeError, ValueError):
        rate = 0.40
    rate = max(0.0, min(0.9, rate))

    # Only the local entropy engine benefits from retrying at a lower drop rate.
    rates = [rate, rate / 2] if mode_lower in ("gemma_entropy", "entropy") else [rate]
    orig_tokens = count_tokens(prompt)

    last_missing: Dict[str, int] = {}
    for r in rates:
        try:
            attempt = _compress_once(prompt, mode_lower, r)
        except Exception as e:  # noqa: BLE001
            print(f"[Compressor] attempt crashed ({e}); falling back")
            continue
        text = attempt.get("compressed_prompt")
        if not isinstance(text, str) or not text.strip():
            continue
        missing = _missing(prompt, text)
        if not missing and count_tokens(text) <= orig_tokens:
            return _finalize(prompt, text, attempt["engine"], r, start)
        last_missing = missing
        print(f"[Compressor] output rejected at drop_rate={r:.3f}: {list(missing)[:8]}")

    # Fallback 1: rule-only output (deterministic and protected)
    try:
        ruled = _rule_strip(prompt)
        if ruled.strip() and not _missing(prompt, ruled) and count_tokens(ruled) <= orig_tokens:
            return _finalize(prompt, ruled, "rule-based (rewrite failed safety check)", 0.0, start)
    except Exception as e:  # noqa: BLE001
        print(f"[Compressor] rule fallback failed ({e})")

    # Fallback 2: return the original untouched
    return _finalize(prompt, prompt, "passthrough (critical-token check failed)", 0.0, start, last_missing)