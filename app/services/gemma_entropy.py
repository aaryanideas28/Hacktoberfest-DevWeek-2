"""
Gemma 2 2B token-entropy / perplexity pruning engine (app/services/gemma_entropy.py).

One forward pass through Gemma 2 2B gives per-token surprisal:
    I(x_{i+1}) = -log P(x_{i+1} | x_{<=i})

Token surprisals are aggregated into WORD scores (mean over the word's tokens),
and the lowest-scoring words are dropped. Dropping whole words (instead of
sub-word tokens) avoids fragments, and rebuilding the text from the original
string keeps spacing and line structure intact.

Anything matched by app.services.protect (negations, numbers, quoted strings,
ALL-CAPS keys, "TAG:" labels, [placeholders], non-ASCII text, code, <data>...</data>)
is NEVER dropped, because "predictable" does not mean "unimportant"
(e.g. "not" after "Do" has very low surprisal but flips the meaning).

No autoregressive generation is performed. Latency is one forward pass:
roughly tens to hundreds of ms on a GPU, much longer on CPU.

Env vars:
    HF_TOKEN                  Hugging Face token (Gemma weights are gated)
    GEMMA_ENTROPY_WINDOW      max tokens per forward pass (default 2048)
"""

import math
import os
import re
import time
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Set, Tuple

from app.services.protect import protected_spans

DEFAULT_MODEL = "google/gemma-2-2b-it"
_WINDOW = int(os.getenv("GEMMA_ENTROPY_WINDOW", "2048"))

# Lazy-loaded globals
_GEMMA_MODEL = None
_GEMMA_TOKENIZER = None
_DEVICE = None
_LOAD_FAILED = False

# Single-entry cache so the safety retry loop in compressor.py does not
# re-run the forward pass for the same prompt at a lower drop rate.
_SCORE_CACHE: Dict[str, Any] = {"key": None, "scores": None}

# Fallback mode: only words that are safe to delete (exact, lowercase match).
SAFE_DROP = {
    "a", "an", "the", "very", "really", "basically", "actually", "kindly", "of",
}


# --------------------------------------------------------------------------- #
# Device / model loading
# --------------------------------------------------------------------------- #
def get_device() -> str:
    """Detect best available compute device (CUDA -> MPS -> CPU)."""
    global _DEVICE
    if _DEVICE is not None:
        return _DEVICE
    try:
        import importlib
        torch = importlib.import_module("torch")

        if torch.cuda.is_available():
            _DEVICE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _DEVICE = "mps"
        else:
            _DEVICE = "cpu"
    except Exception:
        _DEVICE = "cpu"
    return _DEVICE


def load_gemma_local_model(model_name: str = DEFAULT_MODEL):
    """
    Lazy-load Gemma 2 causal LM + tokenizer.
    Returns (model, tokenizer) or (None, None). A failed load is remembered so
    later requests do not pay the retry cost.
    """
    global _GEMMA_MODEL, _GEMMA_TOKENIZER, _LOAD_FAILED
    if _GEMMA_MODEL is not None and _GEMMA_TOKENIZER is not None:
        return _GEMMA_MODEL, _GEMMA_TOKENIZER
    if _LOAD_FAILED:
        return None, None

    try:
        import importlib
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
        AutoModelForCausalLM = transformers.AutoModelForCausalLM
        AutoTokenizer = transformers.AutoTokenizer

        device = get_device()
        hf_token = os.getenv("HF_TOKEN") or None
        print(f"[Gemma Engine] Initializing {model_name} on device: {device}")

        dtype = torch.bfloat16 if device == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="eager",  # recommended for Gemma 2
            token=hf_token,
        )
        model.to(device)
        model.eval()

        _GEMMA_MODEL, _GEMMA_TOKENIZER = model, tokenizer
        return _GEMMA_MODEL, _GEMMA_TOKENIZER
    except Exception as e:  # noqa: BLE001
        _LOAD_FAILED = True
        print(
            f"[Gemma Engine] Local model unavailable ({e}). "
            "Using protected filler-word fallback. "
            "(Gemma is gated: accept the license on Hugging Face and set HF_TOKEN.)"
        )

        return None, None


def reset_model_cache() -> None:
    """Forget a failed load (e.g. after setting HF_TOKEN) without restarting."""
    global _LOAD_FAILED
    _LOAD_FAILED = False


def count_tokens(text: str) -> int:
    """
    Token count using the Gemma tokenizer when it is loaded, else ~4 chars/token.
    Use this for BOTH original and compressed counts so ratios share one unit.
    """
    if not text or not text.strip():
        return 0
    if _GEMMA_TOKENIZER is not None:
        try:
            return len(_GEMMA_TOKENIZER(text, add_special_tokens=False)["input_ids"])
        except Exception:  # noqa: BLE001
            pass
    return max(1, len(text.strip()) // 4)


# --------------------------------------------------------------------------- #
# Word helpers
# --------------------------------------------------------------------------- #
def _word_spans(text: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _protected_flags(words: List[Tuple[int, int]], spans: List[Tuple[int, int]]) -> List[bool]:
    """A word is protected if it overlaps any protected span. Spans are merged+sorted."""
    starts = [s for s, _ in spans]
    flags: List[bool] = []
    for ws, we in words:
        i = bisect_right(starts, we - 1) - 1
        flags.append(i >= 0 and spans[i][1] > ws)
    return flags


def _rebuild(text: str, words: List[Tuple[int, int]], drop: Set[int]) -> str:
    """Rebuild text from the original string, omitting dropped words but keeping layout."""
    out: List[str] = []
    prev_end = 0
    for i, (s, e) in enumerate(words):
        prefix = text[prev_end:s]
        if i in drop:
            if "\n" in prefix:  # keep line breaks even if the word is gone
                out.append(prefix)
        else:
            out.append(prefix)
            out.append(text[s:e])
        prev_end = e
    out.append(text[prev_end:])
    return "".join(out).strip()


# --------------------------------------------------------------------------- #
# Model scoring
# --------------------------------------------------------------------------- #
def _token_surprisals(ids: List[int], model, tokenizer, device: str, window: int) -> List[float]:
    """
    Per-token NLL for every token id. Token 0 (BOS) gets +inf.
    Long inputs are processed in windows (BOS re-prepended) and the NLL is
    computed in row slices to keep the 256k-vocab logits memory bounded.
    """
    import importlib
    torch = importlib.import_module("torch")
    F = importlib.import_module("torch.nn.functional")

    scores = [float("inf")] * len(ids)

    bos = tokenizer.bos_token_id
    start = 0
    while start < len(ids):
        chunk = ids[start:start + window]
        if start > 0 and bos is not None:
            inp, offset = [bos] + chunk, 1
        else:
            inp, offset = chunk, 0
        if len(inp) < 2:
            break

        t = torch.tensor([inp], device=device)
        with torch.no_grad():
            logits = model(input_ids=t).logits[0]  # [L, V]

        parts = []
        for s in range(0, logits.size(0) - 1, 256):
            e = min(s + 256, logits.size(0) - 1)
            parts.append(F.cross_entropy(logits[s:e].float(), t[0, s + 1:e + 1], reduction="none"))
        nll = torch.cat(parts).tolist()  # nll[j] scores inp[j + 1]

        for k in range(len(chunk)):
            inp_idx = k + offset
            if inp_idx >= 1:
                scores[start + k] = nll[inp_idx - 1]
        start += window
    return scores


def _word_scores(prompt: str, words: List[Tuple[int, int]], model, tokenizer, model_name: str) -> List[Optional[float]]:
    """Mean token surprisal per word (None if a word received no scored token)."""
    key = f"{model_name}\x00{prompt}"
    if _SCORE_CACHE["key"] == key:
        return _SCORE_CACHE["scores"]

    device = get_device()
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    tok_scores = _token_surprisals(ids, model, tokenizer, device, _WINDOW)

    starts = [s for s, _ in words]
    sums = [0.0] * len(words)
    counts = [0] * len(words)
    for (cs, ce), sc in zip(offsets, tok_scores):
        if ce <= cs or math.isinf(sc):
            continue  # special tokens / BOS
        last = ce - 1  # last char of a token is never leading whitespace
        wi = bisect_right(starts, last) - 1
        if wi < 0 or last >= words[wi][1]:
            continue  # whitespace-only token
        sums[wi] += sc
        counts[wi] += 1

    result: List[Optional[float]] = [
        (sums[i] / counts[i]) if counts[i] else None for i in range(len(words))
    ]
    _SCORE_CACHE["key"], _SCORE_CACHE["scores"] = key, result
    return result


# --------------------------------------------------------------------------- #
# Result helpers
# --------------------------------------------------------------------------- #
def _result(original: str, compressed: str, start_time: float, engine: str) -> Dict[str, Any]:
    orig_t = count_tokens(original)
    comp_t = count_tokens(compressed)
    saved = max(0, orig_t - comp_t)
    return {
        "compressed_prompt": compressed,
        "original_tokens": orig_t,
        "compressed_tokens": comp_t,
        "tokens_saved": saved,
        "compression_ratio": round((saved / orig_t) * 100, 2) if orig_t > 0 else 0.0,
        "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
        "engine": engine,
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "compressed_prompt": "",
        "original_tokens": 0,
        "compressed_tokens": 0,
        "tokens_saved": 0,
        "compression_ratio": 0.0,
        "latency_ms": 0.0,
        "engine": "none",
    }


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def compress_via_gemma_entropy(
    prompt: str,
    drop_rate: float = 0.40,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """
    Word-level perplexity pruning with Gemma 2 2B.

    Args:
        prompt: raw prompt text.
        drop_rate: fraction of NON-PROTECTED words to drop (0.0 - 0.9).
        model_name: Hugging Face model id.

    Returns dict with: compressed_prompt, original_tokens, compressed_tokens,
    tokens_saved, compression_ratio, latency_ms, engine.
    engine is "gemma-2-2b-forward-pass" or "protected-filler-fallback".
    """
    start_time = time.perf_counter()

    if not prompt or not prompt.strip():
        return _empty_result()

    drop_rate = max(0.0, min(0.9, float(drop_rate)))

    words = _word_spans(prompt)
    if len(words) <= 4:
        return _result(prompt, prompt, start_time, "none (prompt too short)")

    flags = _protected_flags(words, protected_spans(prompt))
    flags[0] = True
    flags[-1] = True

    drop: Optional[Set[int]] = None
    engine = "protected-filler-fallback"

    model, tokenizer = load_gemma_local_model(model_name)
    if model is not None and tokenizer is not None:
        try:
            scores = _word_scores(prompt, words, model, tokenizer, model_name)
            eligible = [i for i in range(len(words)) if not flags[i] and scores[i] is not None]
            k = int(len(eligible) * drop_rate)
            eligible.sort(key=lambda i: scores[i])  # lowest surprisal = most predictable
            drop = set(eligible[:k])
            engine = "gemma-2-2b-forward-pass"
        except Exception as e:  # noqa: BLE001
            print(f"[Gemma Engine] Forward pass failed ({e}); using protected filler fallback.")
            drop = None

    if drop is None:
        # Deterministic fallback: delete only safe articles/intensifiers.
        # drop_rate is ignored here on purpose; length-based scoring is what
        # used to delete "not", digits and "TAG:".
        drop = {
            i for i, (s, e) in enumerate(words)
            if not flags[i] and prompt[s:e] in SAFE_DROP
        }

    compressed = _rebuild(prompt, words, drop)
    return _result(prompt, compressed, start_time, engine)