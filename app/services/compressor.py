import re
from typing import Dict, Any, Optional
from app.services.gemma_entropy import compress_via_gemma_entropy

FILLER_PATTERNS = [
    r"\bcan you please\b", r"\bplease\b", r"\bcould you\b",
    r"\bi would like you to\b", r"\bi want you to\b", r"\bhelp me to\b",
    r"\bas an ai language model\b", r"\bthank you\b", r"\bthanks\b",
    r"\bi am wondering if you could\b", r"\bkindly\b"
]

def estimate_tokens(text: str) -> int:
    """Heuristic estimation (~4 characters per token)."""
    return max(1, len(text.strip()) // 4) if text.strip() else 0

def compress_prompt_rule_based(prompt: str) -> Dict[str, Any]:
    """
    Deterministic rule-based compression: strips conversational filler phrases
    while strictly protecting fenced code blocks.
    """
    original_tokens = estimate_tokens(prompt)
    if not prompt.strip():
        return {
            "compressed": "",
            "compressed_prompt": "",
            "original_tokens": 0,
            "compressed_tokens": 0,
            "tokens_saved": 0,
            "compression_ratio": 0.0,
            "engine": "rule-based"
        }
    
    # Protect fenced code blocks
    code_blocks = re.findall(r"```[\s\S]*?```", prompt)
    placeholder_map = {}
    
    temp_prompt = prompt
    for idx, block in enumerate(code_blocks):
        key = f"__CODE_BLOCK_{idx}__"
        placeholder_map[key] = block
        temp_prompt = temp_prompt.replace(block, key)

    compressed = temp_prompt
    for pattern in FILLER_PATTERNS:
        compressed = re.sub(pattern, "", compressed, flags=re.IGNORECASE)

    for key, block in placeholder_map.items():
        compressed = compressed.replace(key, block)

    compressed = re.sub(r"\s+", " ", compressed).strip()
    compressed_tokens = estimate_tokens(compressed)
    saved = max(0, original_tokens - compressed_tokens)

    return {
        "compressed": compressed,
        "compressed_prompt": compressed,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": saved,
        "compression_ratio": round((saved / original_tokens) * 100, 2) if original_tokens > 0 else 0.0,
        "engine": "rule-based"
    }

def compress_prompt(
    prompt: str,
    mode: str = "hybrid",
    drop_rate: float = 0.40,
) -> Dict[str, Any]:
    """
    Unified prompt compression pipeline supporting:
    - 'gemma_entropy': Gemma 2 2B single forward-pass perplexity/surprisal masking (bottom 40% low-information token drop)
    - 'rule': Regex filler phrase stripper
    - 'hybrid': Rule-based pass followed by Gemma 2 2B token-level entropy pruning
    """
    mode_lower = (mode or "hybrid").lower()

    if mode_lower in ["gemma_entropy", "entropy"]:
        return compress_via_gemma_entropy(prompt, drop_rate=drop_rate)

    if mode_lower == "rule":
        return compress_prompt_rule_based(prompt)

    # Hybrid mode: rule-based cleaning + Gemma 2 2B perplexity pruning
    rule_res = compress_prompt_rule_based(prompt)
    entropy_res = compress_via_gemma_entropy(rule_res["compressed_prompt"], drop_rate=drop_rate)

    orig_tokens = rule_res["original_tokens"]
    comp_tokens = entropy_res["compressed_tokens"]
    saved = max(0, orig_tokens - comp_tokens)

    return {
        "compressed": entropy_res["compressed_prompt"],
        "compressed_prompt": entropy_res["compressed_prompt"],
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "tokens_saved": saved,
        "compression_ratio": round((saved / orig_tokens) * 100, 2) if orig_tokens > 0 else 0.0,
        "latency_ms": entropy_res.get("latency_ms", 0.0),
        "engine": f"hybrid (rule + {entropy_res.get('engine', 'gemma')})",
    }