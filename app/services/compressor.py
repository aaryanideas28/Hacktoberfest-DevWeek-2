"""
Unified prompt compression pipeline (app/services/compressor.py).

Modes:
    "rule"          regex filler-phrase stripper (protected spans are masked first)
    "gemma_entropy" Gemma 2 2B word-level surprisal pruning (alias: "entropy")
    "hybrid"        rule pass, then Gemma entropy pruning (default)

Safety net: after every compression the output is checked with
protect.missing_critical(). If any negation / number / quoted string / label
was lost, the pipeline retries with half and then a quarter of the drop rate,
and finally returns the original prompt unchanged (engine = "passthrough").

Tip: wrap customer emails / payloads in <data>...</data> so they are never
touched; only the instruction text is compressed.
"""

import re
import time
from typing import Any, Dict, Optional

from app.services.gemma_entropy import compress_via_gemma_entropy, count_tokens
from app.services.protect import mask, missing_critical, tidy, unmask

FILLER_PATTERNS = [
    r"\bcan you please\b", r"\bplease\b", r"\bcould you\b",
    r"\bi would like you to\b", r"\bi want you to\b", r"\bhelp me to\b",
    r"\bas an ai language model\b", r"\bthank you\b", r"\bthanks\b",
    r"\bi am wondering if you could\b", r"\bkindly\b",
]


def estimate_tokens(text: str) -> int:
    """Heuristic estimation (~4 characters per token). Kept for backward compatibility."""
    return max(1, len(text.strip()) // 4) if text.strip() else 0


# --------------------------------------------------------------------------- #
# Rule-based stage
# --------------------------------------------------------------------------- #
def _rule_strip(prompt: str) -> str:
    """Strip filler phrases while protected spans (code, quotes, numbers, <data>...) are masked."""
    masked, store = mask(prompt)
    for pattern in FILLER_PATTERNS:
        masked = re.sub(pattern + r"[ \t]*", "", masked, flags=re.IGNORECASE)  # also eat trailing space
    masked = tidy(masked)
    return unmask(masked, store)


def compress_prompt_rule_based(prompt: str) -> Dict[str, Any]:
    """Deterministic regex compression that never touches protected spans."""
    start = time.perf_counter()
    if not prompt or not prompt.strip():
        return {
            "compressed": "",
            "compressed_prompt": "",
            "original_tokens": 0,
            "compressed_tokens": 0,
            "tokens_saved": 0,
            "compression_ratio": 0.0,
            "latency_ms": 0.0,
            "engine": "rule-based",
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
# Single attempt + final packaging
# --------------------------------------------------------------------------- #
def _compress_once(prompt: str, mode: str, rate: float) -> Dict[str, str]:
    """One compression attempt. Returns {"compressed_prompt", "engine"}."""
    if mode in ("gemma_entropy", "entropy"):
        res = compress_via_gemma_entropy(prompt, drop_rate=rate)
        return {"compressed_prompt": res["compressed_prompt"], "engine": res["engine"]}

    if mode == "rule":
        return {"compressed_prompt": _rule_strip(prompt), "engine": "rule-based"}

    # hybrid: rule pass, then entropy pruning on the rule output
    ruled = _rule_strip(prompt)
    res = compress_via_gemma_entropy(ruled, drop_rate=rate)
    return {
        "compressed_prompt": res["compressed_prompt"],
        "engine": f"hybrid (rule + {res['engine']})",
    }


def _finalize(
    original: str,
    compressed: str,
    engine: str,
    rate_used: float,
    start: float,
    missing: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    # Count AFTER any model load so original and compressed share the same unit.
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


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def compress_prompt(
    prompt: str,
    mode: str = "hybrid",
    drop_rate: float = 0.40,
) -> Dict[str, Any]:
    """
    Unified prompt compression with an invariant check.

    - 'gemma_entropy' / 'entropy': Gemma 2 2B single forward-pass word-level pruning
    - 'rule': regex filler stripper
    - 'hybrid' (default): rule pass followed by Gemma entropy pruning

    Returns dict with: compressed, compressed_prompt, original_tokens,
    compressed_tokens, tokens_saved, compression_ratio, latency_ms, engine,
    drop_rate_used, critical_missing.
    """
    start = time.perf_counter()
    mode_lower = (mode or "hybrid").lower()

    if not prompt or not prompt.strip():
        return {
            "compressed": "",
            "compressed_prompt": "",
            "original_tokens": 0,
            "compressed_tokens": 0,
            "tokens_saved": 0,
            "compression_ratio": 0.0,
            "latency_ms": 0.0,
            "engine": "none",
            "drop_rate_used": 0.0,
            "critical_missing": {},
        }

    rate = max(0.0, min(0.9, float(drop_rate if drop_rate is not None else 0.40)))
    rates = [rate] if mode_lower == "rule" else [rate, rate / 2, rate / 4]

    last_missing: Dict[str, int] = {}
    for r in rates:
        attempt = _compress_once(prompt, mode_lower, r)
        missing = missing_critical(prompt, attempt["compressed_prompt"])
        if not missing:
            return _finalize(prompt, attempt["compressed_prompt"], attempt["engine"], r, start)
        last_missing = missing
        print(f"[Compressor] Critical tokens lost at drop_rate={r:.3f}: {list(missing)[:8]}; retrying.")

    return _finalize(
        prompt,
        prompt,
        "passthrough (critical-token check failed)",
        0.0,
        start,
        last_missing,
    )