"""
Context extraction + handoff formatting (app/services/extractor.py).

Pure functions, no network. Used by the handoff router and by app/services/gemma.py.

Schema of a context dict (every key is always present after normalize_context):
    project_summary   str
    tech_stack        list[str]
    decisions_made    list[str]
    completed_work    list[str]
    constraints       list[str]
    failed_approaches list[str]
    open_issues       list[str]
    key_files         list[str]   (file paths, endpoints)
    next_step         str

Trust model: the LLM result is never used as-is. validate_and_merge() drops items that
cannot be found in the transcript (invented numbers / words), then re-adds deterministic
facts (tech stack, file names, hard constraints) that the LLM may have missed or softened.
"""

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

LIST_KEYS = (
    "tech_stack",
    "decisions_made",
    "completed_work",
    "constraints",
    "failed_approaches",
    "open_issues",
    "key_files",
)
STR_KEYS = ("project_summary", "next_step")
MAX_ITEMS = 25
MAX_ITEM_CHARS = 400
NO_NEXT_STEP = "Not stated in the conversation."


# --------------------------------------------------------------------------- #
# Legacy baseline extractor (kept exactly for backwards compatibility with tests)
# --------------------------------------------------------------------------- #
def extract_items(conversation: str) -> Dict[str, List[str]]:
    """
    Baseline heuristic extractor preserved for backwards compatibility with tests.
    """
    decisions = []
    completed_work = []
    unresolved_issues = []
    next_steps = []

    sentences = [
        sentence.strip()
        for sentence in conversation.replace("\n", " ").split(".")
        if sentence.strip()
    ]

    for sentence in sentences:
        text = sentence.lower()

        if "decided to" in text or "we chose" in text:
            decisions.append(sentence)

        is_negative_completion = any(
            phrase in text
            for phrase in [
                "not completed",
                "isn't completed",
                "is not finished",
                "isn't finished",
                "not finished",
            ]
        )

        if (
            ("completed" in text or "finished" in text)
            and not is_negative_completion
        ):
            completed_work.append(sentence)

        if (
            "pending" in text
            or "not working" in text
            or "unresolved" in text
            or "still need to fix" in text
            or is_negative_completion
        ):
            unresolved_issues.append(sentence)

        if (
            "next" in text
            or "need to implement" in text
            or "should implement" in text
            or "todo" in text
        ):
            next_steps.append(sentence)

    return {
        "decisions": decisions,
        "completed_work": completed_work,
        "unresolved_issues": unresolved_issues,
        "next_steps": next_steps,
    }


# --------------------------------------------------------------------------- #
# Small text helpers
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9_]+")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_CRIT = re.compile(r'\d+(?:[.,]\d+)*|"[^"]+"|\b[A-Z]{2,}\b')
_BULLET = re.compile(r"^\s*(?:[-*\u2022]+|\d+[.)])\s+")
_ROLE_PREFIX = re.compile(
    r"^\s*(?:user|assistant|human|ai|claude|chatgpt|gpt|gemini|me|you|system)\s*[:>]\s*", re.I
)
_CODE_FENCE = re.compile(r"```.*?```", re.S)
# Split on ". " / "! " / "? " followed by a new sentence, but NOT inside file names or versions
# such as "main.py" or "v2.0".
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[`/])")

_STOP = frozenset(
    "a an the and or of to in on at for with is are was were be been it its this that these those "
    "as by from so you we i our your my me do does did will would should can could have has had".split()
)
_EMPTY = {
    "", "none", "n/a", "na", "null", "nil", "-", "none reported",
    "none explicitly reported", "not stated", "unknown",
}


def _tokset(s: str) -> set:
    return set(_WORD.findall(s.lower()))


def _clean_item(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, dict):
        x = "; ".join(f"{k}: {v}" for k, v in x.items())
    elif isinstance(x, (list, tuple)):
        x = ", ".join(str(i) for i in x)
    s = " ".join(str(x).split())
    s = _BULLET.sub("", s).strip()
    return s[:MAX_ITEM_CHARS]


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        parts = v.split("\n")
    elif isinstance(v, (list, tuple, set)):
        parts = list(v)
    else:
        parts = [v]
    out = []
    for p in parts:
        s = _clean_item(p)
        if s.lower().rstrip(".") in _EMPTY:
            continue
        out.append(s)
    return out


def _first_str(v: Any) -> str:
    items = _as_list(v)
    return items[0] if items else ""


def _near_dup(a: str, b: str) -> bool:
    if a.lower() == b.lower():
        return True
    ta, tb = _tokset(a), _tokset(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.85


def _union(base: Iterable[str], extra: Iterable[str]) -> List[str]:
    out = list(base)
    for e in extra:
        if not any(_near_dup(e, o) for o in out):
            out.append(e)
    return out[:MAX_ITEMS]


def empty_context() -> Dict[str, Any]:
    d: Dict[str, Any] = {k: [] for k in LIST_KEYS}
    d.update({k: "" for k in STR_KEYS})
    return d


_ALIASES = {
    "decisions": "decisions_made",
    "completed": "completed_work",
    "unresolved_issues": "open_issues",
    "issues": "open_issues",
    "summary": "project_summary",
    "files": "key_files",
    "next_steps": "next_step",
    "failed": "failed_approaches",
}


def normalize_context(data: Any) -> Dict[str, Any]:
    """Coerce anything (LLM JSON, old DB row) into the full schema with correct types."""
    data = data if isinstance(data, dict) else {}
    fixed: Dict[str, Any] = {}
    for k, v in data.items():
        key = str(k).strip().lower()
        fixed[_ALIASES.get(key, key)] = v
    out = empty_context()
    for k in LIST_KEYS:
        out[k] = _union([], _as_list(fixed.get(k)))
    for k in STR_KEYS:
        out[k] = _first_str(fixed.get(k))
    return out


def merge_contexts(parts: Iterable[Any]) -> Dict[str, Any]:
    """Merge per-chunk results of a long transcript. Later chunks win for next_step."""
    out = empty_context()
    for raw in parts:
        if not raw:
            continue
        p = normalize_context(raw)
        for k in LIST_KEYS:
            out[k] = _union(out[k], p[k])
        if not out["project_summary"]:
            out["project_summary"] = p["project_summary"]
        if p["next_step"]:
            out["next_step"] = p["next_step"]
    return out


def is_empty_context(data: Dict[str, Any]) -> bool:
    return not any(data.get(k) for k in (*LIST_KEYS, *STR_KEYS))


# --------------------------------------------------------------------------- #
# Tolerant JSON parsing (LLM output)
# --------------------------------------------------------------------------- #
def parse_json_object(text: Any) -> Optional[Dict[str, Any]]:
    """
    Parse a JSON object out of model output: handles ```json fences, chatty prefixes/suffixes
    and trailing text. Returns None if no valid object is found.
    """
    if not isinstance(text, str):
        return None
    t = text.strip()
    t = re.sub(r"^```[\w-]*\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else None
    except (ValueError, TypeError):
        pass

    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(t[start:i + 1])
                        if isinstance(v, dict):
                            return v
                    except (ValueError, TypeError):
                        pass
                    break
        start = t.find("{", start + 1)
    return None


# --------------------------------------------------------------------------- #
# Rule-based extraction (always available, no API)
# --------------------------------------------------------------------------- #
def split_sentences(text: str) -> List[str]:
    """Sentence splitter that ignores code fences and keeps file names / versions intact."""
    text = _CODE_FENCE.sub("\n", text or "")
    out: List[str] = []
    for line in text.splitlines():
        line = _ROLE_PREFIX.sub("", line.strip())
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        for s in _SENT_END.split(line):
            s = s.strip()
            if len(s) >= 3:
                out.append(s)
    return out


_TECH = [
    ("Python", r"python"), ("FastAPI", r"fastapi"), ("Flask", r"flask"), ("Django", r"django"),
    ("SQLite", r"sqlite3?"), ("PostgreSQL", r"postgres(?:ql)?"), ("MySQL", r"mysql"),
    ("MongoDB", r"mongo(?:db)?"), ("Redis", r"redis"), ("SQLAlchemy", r"sqlalchemy"),
    ("Pydantic", r"pydantic"), ("Pytest", r"pytest"), ("React", r"react(?:\.js)?"),
    ("Vue", r"vue(?:\.js)?"), ("Angular", r"angular"), ("Next.js", r"next\.js|nextjs"),
    ("Node.js", r"node\.js|nodejs"), ("Express", r"express\.js|expressjs"),
    ("TypeScript", r"typescript"), ("JavaScript", r"javascript"), ("HTML", r"html5?"),
    ("CSS", r"css3?"), ("Tailwind CSS", r"tailwind(?:css)?"), ("Docker", r"docker"),
    ("Kubernetes", r"kubernetes|k8s"), ("AWS", r"aws"), ("GCP", r"gcp|google cloud"),
    ("Azure", r"azure"), ("Sentry", r"sentry"), ("PyTorch", r"pytorch|torch"),
    ("OpenAI API", r"openai"), ("Gemini", r"gemini"), ("Gemma", r"gemma(?:[- ]?\d+)?"),
    ("OpenRouter", r"openrouter"), ("LangChain", r"langchain"), ("GraphQL", r"graphql"),
    ("Kafka", r"kafka"), ("RabbitMQ", r"rabbitmq"), ("Nginx", r"nginx"), ("Vite", r"vite"),
    ("Webpack", r"webpack"), ("Java", r"java(?!script)"), ("Rust", r"rust"),
    ("Kotlin", r"kotlin"), ("Swift", r"swift"), ("Flutter", r"flutter"),
]
_TECH_RX = [(name, re.compile(r"\b(?:%s)\b" % pat, re.I)) for name, pat in _TECH]


def detect_tech_stack(text: str) -> List[str]:
    """Technologies actually mentioned in the text, in order of first appearance."""
    found = []
    for name, rx in _TECH_RX:
        m = rx.search(text)
        if m:
            found.append((m.start(), name))
    found.sort()
    return [n for _, n in found][:MAX_ITEMS]


_FILE_RX = re.compile(
    r"(?<![\w/@.-])(?:[\w.-]+/)*[\w.-]+\.(?:py|jsx?|tsx?|html|css|json|ya?ml|toml|md|txt|sql|db|env|sh|ini|cfg|go|rs|java|kt|cs|cpp|rb|php)\b",
    re.I,
)
_DOTENV_RX = re.compile(r"(?<![\w/])\.env(?:\.\w+)?\b")
_ENDPOINT_RX = re.compile(r"(?<![\w/])/(?:api|v\d+)/[\w/{}.-]*\w")
_NOT_FILES = {"node.js", "next.js", "vue.js", "react.js", "express.js"}


def detect_key_files(text: str) -> List[str]:
    """File paths and API endpoints mentioned in the text (exact strings, in order)."""
    hits = []
    for rx in (_FILE_RX, _DOTENV_RX, _ENDPOINT_RX):
        for m in rx.finditer(text):
            s = m.group().rstrip(".,;:")
            if s.lower() in _NOT_FILES:
                continue
            hits.append((m.start(), s))
    hits.sort()
    seen, out = set(), []
    for _, s in hits:
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out[:20]


# Strong constraint wording always counts; weak wording only if the sentence is not a decision.
_CONSTRAINT_STRONG = re.compile(
    r"\b(?:do not|don'?t|never|must|should not|shouldn'?t)\b", re.I
)
_CONSTRAINT_WEAK = re.compile(
    r"\b(?:keep|required?|requirements?|constraints?|exactly|at most|at least|no more than|"
    r"make sure|ensure|avoid)\b",
    re.I,
)
_DECISION = re.compile(
    r"\b(?:decided|decide to|chose|chosen|selected|settled on|going with|go with|switched to|"
    r"agreed|will use|we'?ll use|we use|opted)\b",
    re.I,
)
_COMPLETED = re.compile(
    r"\b(?:completed|finished|done|implemented|added|fixed|resolved|merged|deployed|created|"
    r"built|working now|now works|now working)\b",
    re.I,
)
_NEGATED = re.compile(
    r"\b(?:not|never|isn'?t|aren'?t|wasn'?t|weren'?t|hasn'?t|haven'?t|hadn'?t|didn'?t|doesn'?t|"
    r"don'?t|yet|still|unfinished|need(?:s)? to|todo|should|will|would|going to|let'?s|plan to)\b",
    re.I,
)
_FAILED = re.compile(
    r"\b(?:tried|attempted|failed|didn'?t work|did not work|doesn'?t work|does not work|"
    r"abandon(?:ed)?|dropped|ditched|rejected|ruled out|deprecated|gave up|reverted|rolled back|"
    r"too slow|dead end)\b",
    re.I,
)
_OPEN = re.compile(
    r"\b(?:not working|isn'?t working|still|pending|unresolved|open issue|bug|broken|error|"
    r"exception|traceback|crash(?:es|ed)?|fails|failing|issue|problem|blocked|stuck|"
    r"not finished|not completed|isn'?t completed|unfinished)\b",
    re.I,
)
_NEXT = re.compile(
    r"\b(?:next|todo|to-do|to do|need to|needs to|should implement|will implement|plan to|"
    r"remaining|follow-up|going to|let'?s|then we|after that)\b",
    re.I,
)


def extract_structured_context(transcript: str) -> dict:
    """
    Deterministic extractor. A sentence can land in several lists (a decision that is also a
    constraint), questions are skipped, and next_step is the MOST RECENT next-step sentence.
    Nothing is invented: empty lists stay empty.
    """
    transcript = transcript or ""
    prose = _CODE_FENCE.sub(" ", transcript)

    out = empty_context()
    decisions, constraints, completed = [], [], []
    failed, open_issues, next_steps = [], [], []

    for s in split_sentences(transcript):
        if s.endswith("?"):
            continue

        is_decision = bool(_DECISION.search(s))
        if _CONSTRAINT_STRONG.search(s) or (not is_decision and _CONSTRAINT_WEAK.search(s)):
            constraints.append(s)
        if is_decision:
            decisions.append(s)

        is_completed = bool(_COMPLETED.search(s)) and not _NEGATED.search(s)
        if is_completed:
            completed.append(s)
        elif _FAILED.search(s):
            failed.append(s)
        elif _OPEN.search(s):
            open_issues.append(s)

        if not is_completed and _NEXT.search(s):
            next_steps.append(s)

    out["decisions_made"] = _union([], _as_list(decisions))
    out["constraints"] = _union([], _as_list(constraints))
    out["completed_work"] = _union([], _as_list(completed))
    out["failed_approaches"] = _union([], _as_list(failed))
    out["open_issues"] = _union([], _as_list(open_issues))
    out["tech_stack"] = detect_tech_stack(prose)
    out["key_files"] = detect_key_files(prose)
    out["next_step"] = _clean_item(next_steps[-1]) if next_steps else ""
    return out


# --------------------------------------------------------------------------- #
# Verify an LLM extraction against the transcript, then merge deterministic facts
# --------------------------------------------------------------------------- #
def _grounded(item: str, t_words: set, t_nums: set, min_frac: float = 0.5) -> bool:
    """An item is grounded if its numbers exist in the transcript and most content words do."""
    if any(n not in t_nums for n in _NUM.findall(item)):
        return False
    words = [
        w for w in _WORD.findall(item.lower())
        if len(w) > 2 and w not in _STOP and not w.isdigit()
    ]
    if not words:
        return True
    return sum(w in t_words for w in words) / len(words) >= min_frac


def _covered(item: str, existing: List[str], thr: float = 0.5) -> bool:
    """True if `existing` already expresses `item` (same numbers/labels and enough shared words)."""
    joined = " ".join(existing).lower()
    if any(tok.lower() not in joined for tok in _CRIT.findall(item)):
        return False
    ta = {w for w in _tokset(item) if w not in _STOP}
    if not ta:
        return True
    return any(len(ta & _tokset(e)) / len(ta) >= thr for e in existing)


def validate_and_merge(
    llm_data: Any, rule_data: Dict[str, Any], transcript: str
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (final_context, notes). Never trusts the LLM:
      1. drop items whose numbers / words are not in the transcript
      2. re-add rule-extracted constraints the LLM missed or softened
      3. union in deterministic tech stack and file names
      4. fall back to the rule-based next_step if the LLM's is missing or ungrounded
    """
    data = normalize_context(llm_data)
    rule = normalize_context(rule_data)
    notes: List[str] = []

    t_low = transcript.lower()
    t_words = set(_WORD.findall(t_low))
    t_nums = set(_NUM.findall(transcript))

    dropped = 0
    for key in ("decisions_made", "completed_work", "constraints", "failed_approaches", "open_issues"):
        kept = [i for i in data[key] if _grounded(i, t_words, t_nums)]
        dropped += len(data[key]) - len(kept)
        data[key] = kept

    kept_tech = [t for t in data["tech_stack"] if all(w in t_words for w in _WORD.findall(t.lower()))]
    dropped += len(data["tech_stack"]) - len(kept_tech)
    data["tech_stack"] = kept_tech

    kept_files = [f for f in data["key_files"] if f.lower() in t_low]
    dropped += len(data["key_files"]) - len(kept_files)
    data["key_files"] = kept_files

    if data["project_summary"] and not _grounded(data["project_summary"], t_words, t_nums, 0.4):
        data["project_summary"] = ""
        dropped += 1

    if not data["next_step"] or not _grounded(data["next_step"], t_words, t_nums):
        data["next_step"] = rule["next_step"]

    data["tech_stack"] = _union(
        data["tech_stack"], [t for t in rule["tech_stack"] if not _covered(t, data["tech_stack"], 1.0)]
    )
    data["key_files"] = _union(data["key_files"], rule["key_files"])

    missing_constraints = [
        c for c in rule["constraints"] if not _covered(c, data["constraints"])
    ][:10]
    data["constraints"] = _union(data["constraints"], missing_constraints)

    if dropped:
        notes.append(
            f"{dropped} AI-extracted item(s) were removed because they could not be verified "
            "against the transcript."
        )
    if missing_constraints:
        notes.append(
            f"{len(missing_constraints)} constraint(s) were added from the transcript that the "
            "AI summary missed."
        )
    return data, notes


# --------------------------------------------------------------------------- #
# Markdown handoff artifact
# --------------------------------------------------------------------------- #
def _one_line(s: Any) -> str:
    return " ".join(str(s).split())


def _section(title: str, items: Any, always: bool = False) -> str:
    items = _as_list(items)
    if not items:
        return f"### {title}\n- None reported\n" if always else ""
    return f"### {title}\n" + "\n".join(f"- {_one_line(i)}" for i in items) + "\n"


def format_handoff_markdown(project_name: str, data: dict = None) -> str:
    """
    Build the paste-into-a-new-chat artifact. Supports both
    format_handoff_markdown(project_name, data) and format_handoff_markdown(data, project_name).
    """
    if isinstance(project_name, dict) and isinstance(data, str):
        data, project_name = project_name, data
    data = normalize_context(data or {})

    summary = data["project_summary"]
    next_step = data["next_step"] or NO_NEXT_STEP

    parts = [
        "# 🔄 ContextSwitch Handoff Artifact",
        f"**Project:** {_one_line(project_name)}",
        "",
        "> You are continuing an in-progress project from a previous conversation. Treat the "
        "context below as established fact. Do not ask for it again, and do not break a "
        "constraint without asking first.",
        "",
    ]
    if summary:
        parts += ["### 📌 Project Summary", summary, ""]
    sections = [
        _section("🧱 Tech Stack", data["tech_stack"]),
        _section("🎯 Confirmed Decisions", data["decisions_made"], always=True),
        _section("✅ Completed Work", data["completed_work"]),
        _section("⚠️ Strict Constraints (DO NOT ALTER)", data["constraints"], always=True),
        _section("❌ Discarded / Failed Approaches", data["failed_approaches"], always=True),
        _section("🐞 Open Issues", data["open_issues"]),
        _section("📂 Key Files & Endpoints", data["key_files"]),
    ]
    parts += [s for s in sections if s]
    parts += ["### 🚀 Immediate Next Task", f"- {_one_line(next_step)}", ""]
    return "\n".join(parts)


# Aliases for handoff router
extract_context_items = extract_structured_context
format_markdown_artifact = format_handoff_markdown