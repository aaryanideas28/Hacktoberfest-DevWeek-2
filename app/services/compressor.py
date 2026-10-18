import re

FILLER_PATTERNS = [
    r"\bcan you please\b", r"\bplease\b", r"\bcould you\b",
    r"\bi would like you to\b", r"\bi want you to\b", r"\bhelp me to\b",
    r"\bas an ai language model\b", r"\bthank you\b", r"\bthanks\b"
]

def estimate_tokens(text: str) -> int:
    """Heuristic estimation (~4 characters per token)."""
    return len(text.strip()) // 4

def compress_prompt_rule_based(prompt: str) -> dict:
    original_tokens = estimate_tokens(prompt)
    
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

    return {
        "compressed": compressed,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": max(0, original_tokens - compressed_tokens)
    }