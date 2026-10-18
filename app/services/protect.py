"""
Critical-token protection layer (app/services/protect.py).

Both compression engines (regex filler stripper and Gemma entropy pruning) must
never delete tokens that carry hard constraints: negations, numbers, quoted
strings, ALL-CAPS keys, "TAG:"-style labels, placeholders, non-ASCII text, code,
and anything wrapped in <data>...</data>.

Public API:
    protected_spans(text)            -> merged, sorted [(start, end)] char spans
    mask(text) / unmask(text, store) -> swap protected spans for placeholders and back
    tidy(text)                       -> whitespace cleanup (run on MASKED text only)
    missing_critical(orig, comp)     -> {token: count_lost}; empty dict means safe
"""

import re
from collections import Counter
from typing import Dict, List, Tuple

# Edit this tuple to tune how conservative the compressor is.
CRITICAL_WORDS = (
    "not", "no", "never", "none", "nor", "without", "cannot", "cant", "dont",
    "wont", "only", "exactly", "except", "exept", "must", "always", "once",
    "twice", "off", "under", "over", "than", "unless", "until", "before",
    "after", "first", "second", "third", "last", "every", "each", "all", "any",
    "one", "two", "three", "max", "min", "maximum", "minimum", "most", "least",
    "if",
)

_AUX = r"(?:do|does|did|can|could|would|should|must|will|is|are|was|were|has|have)"

_PATTERNS = [
    r"```[\s\S]*?```",                       # fenced code
    r"`[^`\n]+`",                            # inline code
    r"<data>[\s\S]*?</data>",                # payloads (customer emails etc.)
    r'"[^"\n]{1,300}"',                      # "quoted strings" stay verbatim
    r"\u201c[^\u201d\n]{1,300}\u201d",       # curly-quoted strings
    r"\[[^\]\n]{1,80}\]",                    # [placeholders]
    r"https?://\S+",                         # URLs
    r"[\w.+-]+@[\w-]+\.[\w.-]+",             # emails
    r"[^\x00-\x7F]+",                        # Hindi, emoji, curly punctuation, etc.
    r"\b[A-Z][A-Za-z]*(?: \d+)?:",           # TAG:  From:  EMAIL 1:  Formating:
    r"\b[A-Z]{2,}\b",                        # SAFETY, ONE, EMI, CEO, FIRST
    r"\d[\d,.\-/]*\w*",                      # 3, 120, 7kW, 1800-555-0142
    rf"(?i:\b{_AUX}\s+not\b)",               # "do not", "must not", ...
    r"(?i:\bat\s+(?:most|least)\b)",         # "at most", "at least"
    r"(?i:\b\w+n['\u2019]t\b)",              # don't, can't, isn't, ...
    r"(?i:\b(?:" + "|".join(CRITICAL_WORDS) + r")\b)",
]

_RX = re.compile("|".join(f"(?:{p})" for p in _PATTERNS))
_PLACEHOLDER_RX = re.compile("\u27e6(\\d+)\u27e7")


def protected_spans(text: str) -> List[Tuple[int, int]]:
    """Merged, sorted, non-overlapping character spans that must be preserved."""
    spans = sorted(m.span() for m in _RX.finditer(text))
    merged: List[Tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def mask(text: str):
    """Replace every protected match with a placeholder. Returns (masked_text, store)."""
    store: List[str] = []

    def repl(m):
        store.append(m.group())
        return f"\u27e6{len(store) - 1}\u27e7"

    return _RX.sub(repl, text), store


def unmask(text: str, store: List[str]) -> str:
    return _PLACEHOLDER_RX.sub(lambda m: store[int(m.group(1))], text)


def tidy(text: str) -> str:
    """
    Whitespace cleanup. Call on MASKED text so code blocks / data payloads are
    never touched. Keeps newlines (line structure matters in prompts).
    """
    text = re.sub(r"(?<=\S)[ \t]{2,}(?=\S)", " ", text)   # interior runs only, keeps indentation
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def missing_critical(original: str, compressed: str) -> Dict[str, int]:
    """
    Critical tokens whose occurrence count dropped during compression.
    Empty dict means every negation / number / quoted string / label survived.
    """
    o = Counter(m.group().lower() for m in _RX.finditer(original))
    c = Counter(m.group().lower() for m in _RX.finditer(compressed))
    return {k: v - c.get(k, 0) for k, v in o.items() if c.get(k, 0) < v}