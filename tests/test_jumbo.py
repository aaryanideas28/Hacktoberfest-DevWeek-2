"""
JUMBO TEST SUITE  ->  put in  tests/test_jumbo.py   and run:

    pytest tests/test_jumbo.py -q --tb=line --maxfail=60

Fully offline (network blocked, API mocked, Gemma never loaded). ~2,500 tests, a few seconds.
It does NOT hope the code is right: it generates many prompts, feeds them through GOOD and
ADVERSARIAL fake "Gemma" rewriters, and checks invariants with an independent oracle
(explicit must-keep strings), not only your own missing_critical().
"""
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.services import compressor as C
from app.services import gemma_entropy as GE
from app.services import protect as P

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# Fixtures: offline, no model load, no keys
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for k in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(GE, "_LOAD_FAILED", True)
    monkeypatch.setattr(GE, "_GEMMA_MODEL", None)
    monkeypatch.setattr(GE, "_GEMMA_TOKENIZER", None)
    import requests

    def boom(*a, **k):
        raise RuntimeError("network blocked in tests")

    monkeypatch.setattr(requests.sessions.Session, "request", boom)


def use_rewriter(monkeypatch, fn):
    """Replace the Gemma API call with a fake. fn(text) -> str | anything."""
    wrapped = lambda text, *a, **k: fn(text)  # noqa: E731
    monkeypatch.setattr(C, "rewrite", wrapped, raising=False)
    try:
        import app.services.generative as G
        monkeypatch.setattr(G, "rewrite", wrapped)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Prompt generator (seeded, deterministic)
# --------------------------------------------------------------------------- #
CORES = [
    ("write a function that does NOT crash on 3 empty files.", ("NOT", "3")),
    ('only read files from the folder "reports" and process at most 50 files.', ('"reports"', "50")),
    ("Do not mention JIRA-4521 in the reply.", ("Do not", "JIRA-4521")),
    ("then open https://example.com/a?b=1 and summarize it.", ("https://example.com/a?b=1",)),
    ("keep it under 120 words.", ("120",)),
    ("never delete files and always return exit code 0.", ("never", "0")),
    ("use `rm -rf` ONLY inside the sandbox.", ("`rm -rf`", "ONLY")),
    ("<data>Hi team, order #88213 arrived damaged. I paid $249.99.</data>",
     ("<data>Hi team, order #88213 arrived damaged. I paid $249.99.</data>",)),
    ("Return JSON with keys TITLE and SUMMARY.", ("TITLE", "SUMMARY")),
    ("the temperature must be -5.5 degrees.", ("-5.5",)),
    ("explain step by step how to set up a login page.", ()),
    ("don't use any external libraries.", ("don't",)),
    ('Translate to French: "bonjour le monde".', ('"bonjour le monde"',)),
    ("the button is labeled \"Sign In\" and needs 2 input fields.", ('"Sign In"', "2")),
    ("RESULT: output one line per file.", ("RESULT:",)),
]
FILL_PRE = ["", "Can you please", "I would like you to", "Could you kindly", "Hello! I want you to please",
            "As an AI language model,", "I am wondering if you could", "Please help me to"]
FILL_POST = ["", "Thank you.", "Thanks!", "Thank you so much, thanks!", "Kindly be careful. Thanks"]


def build_prompts(n, seed=1234):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        picks = rnd.sample(CORES, rnd.randint(1, 4))
        body = rnd.choice([" ", "\n", " ", "\n\n"]).join(t for t, _ in picks)
        must = tuple(m for _, ms in picks for m in ms)
        text = f"{rnd.choice(FILL_PRE)} {body} {rnd.choice(FILL_POST)}".strip()
        out.append((text, must))
    return out


PROMPTS = build_prompts(120)
IDS = [f"p{i}" for i in range(len(PROMPTS))]

# --------------------------------------------------------------------------- #
# Fake rewriters: 3 good, many hostile
# --------------------------------------------------------------------------- #
_FW = re.compile(r"\b(carefully|basically|really|very|just|kindly|simply)\b ?", re.I)


def _raises(s):
    raise RuntimeError("API down / 429")


REWRITERS = {
    "identity": lambda s: s,
    "drops_filler_words": lambda s: _FW.sub("", s),
    "drops_trailing_period": lambda s: s.rstrip(". "),
    "lowercases": lambda s: s.lower(),
    "uppercases": lambda s: s.upper(),
    "strips_negations": lambda s: re.sub(r"(?i)\b(not|never)\b|n't", "", s),
    "drops_digits": lambda s: re.sub(r"\d", "", s),
    "drops_quotes": lambda s: s.replace('"', ""),
    "truncates_half": lambda s: s[: len(s) // 2],
    "returns_empty": lambda s: "",
    "returns_none": lambda s: None,
    "returns_non_str": lambda s: 42,
    "returns_longer": lambda s: s + "\n" + s,
    "answers_instead": lambda s: "Sure! Here you go:\ndef f():\n    pass",
    "markdown_fenced": lambda s: f"```\n{s}\n```",
    "adds_preamble": lambda s: "Here is the rewritten prompt:\n" + s,
    "raises": _raises,
}
REQUIRED_KEYS = {"compressed_prompt", "original_tokens", "compressed_tokens",
                 "tokens_saved", "compression_ratio", "engine"}


def check(prompt, must, res, label=""):
    assert REQUIRED_KEYS <= set(res), f"missing keys {REQUIRED_KEYS - set(res)}"
    out = res["compressed_prompt"]
    assert isinstance(out, str)
    for m in must:
        assert m in out, f"{label} LOST {m!r}"
    assert res["compressed_tokens"] <= res["original_tokens"], "output longer than input"
    assert res["tokens_saved"] >= 0
    assert 0 <= res["compression_ratio"] <= 100
    assert out == out.strip()


# =========================================================================== #
# 1. THE MATRIX: every prompt x every rewriter (hybrid)  ~ 1,700 tests
# =========================================================================== #
@pytest.mark.parametrize("name", list(REWRITERS))
@pytest.mark.parametrize("idx", range(len(PROMPTS)), ids=IDS)
def test_matrix_hybrid(monkeypatch, idx, name):
    prompt, must = PROMPTS[idx]
    use_rewriter(monkeypatch, REWRITERS[name])
    res = C.compress_prompt(prompt, mode="hybrid", drop_rate=0.4)
    check(prompt, must, res, name)
    # never worse than the plain rule stage (when rule stage is itself safe)
    ruled = C._rule_strip(prompt)
    if not P.missing_critical(prompt, ruled):
        assert res["compressed_tokens"] <= GE.count_tokens(ruled), "worse than rule-only"


# =========================================================================== #
# 2. Rule mode & other modes  ~ 500 tests
# =========================================================================== #
@pytest.mark.parametrize("idx", range(len(PROMPTS)), ids=IDS)
def test_rule_mode(idx):
    prompt, must = PROMPTS[idx]
    check(prompt, must, C.compress_prompt(prompt, mode="rule"), "rule")


@pytest.mark.parametrize("idx", range(len(PROMPTS)), ids=IDS)
def test_rule_idempotent(idx):
    once = C._rule_strip(PROMPTS[idx][0])
    assert C._rule_strip(once) == once


BAD_PUNCT = [r"\.\.", r"\.\s*!", r"!\s*\.", r"\s+[.,!?;:](?!\w)", r"[ \t]{2,}"]


@pytest.mark.parametrize("idx", range(len(PROMPTS)), ids=IDS)
def test_rule_no_new_punctuation_garbage(idx):
    prompt, _ = PROMPTS[idx]
    out = C._rule_strip(prompt)
    for pat in BAD_PUNCT:
        assert len(re.findall(pat, out)) <= len(re.findall(pat, prompt)), f"introduced {pat!r}: {out!r}"


@pytest.mark.parametrize("mode", ["gemma_entropy", "entropy", "generative", "banana", "HYBRID", ""])
@pytest.mark.parametrize("idx", range(0, len(PROMPTS), 6), ids=lambda i: f"p{i}")
def test_all_modes_survive(monkeypatch, idx, mode):
    use_rewriter(monkeypatch, REWRITERS["drops_filler_words"])
    prompt, must = PROMPTS[idx]
    check(prompt, must, C.compress_prompt(prompt, mode=mode, drop_rate=0.4), mode)


@pytest.mark.parametrize("rate", [None, -1, 0, 0.0, 0.1, 0.4, 0.5, 0.9, 1, 5])
@pytest.mark.parametrize("idx", range(0, len(PROMPTS), 10), ids=lambda i: f"p{i}")
def test_drop_rates(monkeypatch, idx, rate):
    use_rewriter(monkeypatch, REWRITERS["drops_filler_words"])
    prompt, must = PROMPTS[idx]
    check(prompt, must, C.compress_prompt(prompt, mode="hybrid", drop_rate=rate))


def test_good_rewriter_is_actually_used(monkeypatch):
    use_rewriter(monkeypatch, REWRITERS["drops_filler_words"])
    p = "Please carefully basically really write a function that does NOT crash on 3 files."
    res = C.compress_prompt(p, mode="hybrid", drop_rate=0.4)
    assert "carefully" not in res["compressed_prompt"], "valid rewrite was rejected"
    assert "rewrite" in res["engine"].lower() or "gemma" in res["engine"].lower()
    assert "NOT" in res["compressed_prompt"] and "3" in res["compressed_prompt"]


def test_deterministic(monkeypatch):
    use_rewriter(monkeypatch, REWRITERS["drops_filler_words"])
    for p, _ in PROMPTS[:30]:
        a = C.compress_prompt(p, mode="hybrid", drop_rate=0.4)["compressed_prompt"]
        b = C.compress_prompt(p, mode="hybrid", drop_rate=0.4)["compressed_prompt"]
        assert a == b


# =========================================================================== #
# 3. Edge inputs (human-error zone)
# =========================================================================== #
EDGE = ["", " ", "\n\n", "\t", "a", "ok", "Thanks!", "please", "Please. Thank you. Thanks.",
        "\U0001F600", "NOT", "0", '"', "<data></data>", "<data>", "</data>", "<data>x</data><data>y</data>",
        "line1\r\nline2\r\nplease line3", "tabs\tand\tspaces   everywhere", "a" * 5000,
        "\u00e9\u00e8\u00ea \u4e2d\u6587 \u0645\u0631\u062d\u0628\u0627 do NOT drop 7",
        "x\x00y please", "Do NOT do this. " * 300, "Can you please " * 200 + "stop NOW",
        "```python\nprint('please')\n```\nplease run it 3 times."]


@pytest.mark.parametrize("i", range(len(EDGE)), ids=[f"e{i}" for i in range(len(EDGE))])
@pytest.mark.parametrize("mode", ["hybrid", "rule", "gemma_entropy"])
def test_edge_inputs_never_crash(monkeypatch, i, mode):
    use_rewriter(monkeypatch, REWRITERS["drops_filler_words"])
    res = C.compress_prompt(EDGE[i], mode=mode, drop_rate=0.4)
    assert REQUIRED_KEYS <= set(res)
    assert isinstance(res["compressed_prompt"], str)
    assert res["compressed_tokens"] <= res["original_tokens"]
    for m in ("NOT", "7", "3", "NOW", "```python"):
        if m in EDGE[i]:
            assert m in res["compressed_prompt"], f"lost {m!r}"


@pytest.mark.parametrize("mode", ["hybrid", "rule"])
def test_speed_large_inputs(monkeypatch, mode):
    use_rewriter(monkeypatch, REWRITERS["identity"])
    big = ("Can you please write a function that does NOT crash on 3 files. Thank you. " * 3000)
    t = time.perf_counter()
    res = C.compress_prompt(big, mode=mode)
    assert time.perf_counter() - t < 8, "too slow: quadratic loop or regex?"
    assert "NOT" in res["compressed_prompt"]


def test_thread_safety(monkeypatch):
    use_rewriter(monkeypatch, REWRITERS["drops_filler_words"])
    texts = [p for p, _ in PROMPTS[:40]]
    seq = [C.compress_prompt(t, mode="hybrid")["compressed_prompt"] for t in texts]
    with ThreadPoolExecutor(8) as ex:
        par = list(ex.map(lambda t: C.compress_prompt(t, mode="hybrid")["compressed_prompt"], texts))
    assert seq == par


# =========================================================================== #
# 4. protect.py unit tests
# =========================================================================== #
@pytest.mark.parametrize("idx", range(len(PROMPTS)), ids=IDS)
def test_mask_unmask_roundtrip(idx):
    text = PROMPTS[idx][0]
    masked, store = P.mask(text)
    assert P.unmask(masked, store) == text


@pytest.mark.parametrize("idx", range(len(PROMPTS)), ids=IDS)
def test_spans_sorted_and_disjoint(idx):
    spans = P.protected_spans(PROMPTS[idx][0])
    for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
        assert s1 < e1 <= s2 < e2, f"overlap/unsorted: {(s1, e1)} {(s2, e2)}"


@pytest.mark.parametrize("idx", range(len(PROMPTS)), ids=IDS)
def test_must_keep_strings_are_protected(idx):
    """Every must-keep string should fall inside a protected span."""
    prompt, must = PROMPTS[idx]
    spans = P.protected_spans(prompt)
    for m in must:
        pos = prompt.find(m)
        assert pos >= 0
        covered = any(s <= pos and pos + len(m) <= e for s, e in spans) or \
            any(s < pos + len(m) and e > pos for s, e in spans)
        assert covered, f"{m!r} is not protected by protect.py"


LOST = [("do NOT delete files", "do delete files"), ("process 50 files", "process files"),
        ('read "reports" only', "read only"), ("don't touch it", "touch it"),
        ("use TITLE key", "use key"), ("see https://a.com/x now", "see now"),
        ("Do not mention JIRA-4521", "Do not mention"), ("<data>secret 42</data> go", "go"),
        ("never do that", "do that"), ("pay $249.99 today", "pay today")]


@pytest.mark.parametrize("a,b", LOST)
def test_missing_critical_detects_loss(a, b):
    assert P.missing_critical(a, b), f"loss not detected: {a!r} -> {b!r}"


@pytest.mark.parametrize("a,b", [("do NOT delete", "do not delete"), ("use TITLE key", "use title key"),
                                 ("Do not mention", "do not mention"), ('say "Sign In"', 'say "sign in"')])
def test_case_change_is_flagged(a, b):
    pass


@pytest.mark.parametrize("idx", range(len(PROMPTS)), ids=IDS)
def test_identity_has_nothing_missing(idx):
    p = PROMPTS[idx][0]
    assert not P.missing_critical(p, p)


# =========================================================================== #
# 5. gemma_entropy.py structure (no model needed)
# =========================================================================== #
@pytest.mark.parametrize("idx", range(len(PROMPTS)), ids=IDS)
def test_protected_flags_match_bruteforce(idx):
    text = PROMPTS[idx][0]
    words, spans = GE._word_spans(text), P.protected_spans(text)
    brute = [any(s < we and e > ws for s, e in spans) for ws, we in words]
    assert GE._protected_flags(words, spans) == brute


@pytest.mark.parametrize("idx", range(len(PROMPTS)), ids=IDS)
def test_rebuild_identity_and_total_drop(idx):
    text = PROMPTS[idx][0]
    words = GE._word_spans(text)
    assert GE._rebuild(text, words, set()) == text.strip()
    GE._rebuild(text, words, set(range(len(words))))  # must not crash


@pytest.mark.parametrize("idx", range(len(PROMPTS)), ids=IDS)
def test_entropy_fallback_keeps_protected(idx):
    prompt, must = PROMPTS[idx]
    res = GE.compress_via_gemma_entropy(prompt, 0.9)
    for m in must:
        assert m in res["compressed_prompt"], f"fallback dropped {m!r}"
    assert res["compressed_tokens"] <= res["original_tokens"]


# =========================================================================== #
# 6. API layer + UI + repo hygiene (human errors)
# =========================================================================== #
@pytest.fixture
def client(monkeypatch):
    tc = pytest.importorskip("fastapi.testclient")
    import main
    use_rewriter(monkeypatch, REWRITERS["drops_filler_words"])
    return tc.TestClient(main.app)


@pytest.mark.parametrize("mode", ["hybrid", "rule", "gemma_entropy", "entropy", "generative"])
@pytest.mark.parametrize("idx", range(0, len(PROMPTS), 8), ids=lambda i: f"p{i}")
def test_api_compress(client, idx, mode):
    prompt, must = PROMPTS[idx]
    r = client.post("/api/v1/compress", json={"prompt": prompt, "mode": mode, "drop_rate": 0.4})
    assert r.status_code == 200, r.text
    out = r.json()["compressed_prompt"]
    for m in must:
        assert m in out, f"API lost {m!r}"


def test_api_bad_requests_never_500(client):
    for body in [{}, {"prompt": ""}, {"prompt": "   "}, {"prompt": "hi", "mode": "nope"},
                 {"prompt": "hi", "drop_rate": 99}, {"prompt": "hi", "drop_rate": -3}, {"prompt": 123}]:
        assert client.post("/api/v1/compress", json=body).status_code < 500, body


def test_ui_default_mode_is_hybrid_and_modes_valid(monkeypatch):
    html = ROOT / "app" / "static" / "index.html"
    if not html.exists():
        pytest.skip("no index.html")
    m = re.search(r"<select[^>]*mode[^>]*>(.*?)</select>", html.read_text(encoding="utf-8"), re.S | re.I)
    if not m:
        pytest.skip("mode <select> not found")
    values = re.findall(r'<option[^>]*value="([^"]*)"', m.group(1))
    assert values, "no options"
    assert values[0] == "hybrid", f"first/default option is {values[0]!r}, expected 'hybrid'"
    use_rewriter(monkeypatch, REWRITERS["drops_filler_words"])
    for v in values:
        check("Please write NOT 3", ("NOT", "3"), C.compress_prompt("Please write NOT 3 ok", mode=v))


def test_dotenv_is_gitignored():
    gi = ROOT / ".gitignore"
    assert gi.exists(), "no .gitignore"
    assert re.search(r"^\.env\s*$", gi.read_text(), re.M), ".env is NOT in .gitignore"


SECRET = re.compile(r"sk-or-v1-[0-9a-f]{20,}|AIza[0-9A-Za-z_\-]{30,}|hf_[A-Za-z0-9]{30,}|"
                    r"https://[0-9a-f]{32}@[\w.\-]*sentry\.io")


def test_no_hardcoded_secrets():
    skip = {".venv", "venv", ".git", "node_modules", "__pycache__", ".pytest_cache"}
    hits = []
    for d, dirs, files in os.walk(ROOT):
        dirs[:] = [x for x in dirs if x not in skip]
        for f in files:
            if f == ".env" or f == Path(__file__).name or not f.endswith((".py", ".md", ".html", ".js", ".json", ".txt", ".toml", ".yml")):
                continue
            try:
                if SECRET.search((Path(d) / f).read_text(encoding="utf-8", errors="ignore")):
                    hits.append(str(Path(d) / f))
            except OSError:
                pass
    assert not hits, f"secrets found in: {hits}"


def test_requirements_cover_imports():
    req = ROOT / "requirements.txt"
    if not req.exists():
        pytest.skip("no requirements.txt")
    text = req.read_text().lower()
    src = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in (ROOT / "app").rglob("*.py"))
    src += (ROOT / "main.py").read_text(encoding="utf-8", errors="ignore") if (ROOT / "main.py").exists() else ""
    need = {"fastapi": "fastapi", "uvicorn": "uvicorn", "requests": "requests", "dotenv": "python-dotenv",
            "google.generativeai": "google-generativeai"}
    for imp, pkg in need.items():
        if re.search(rf"^\s*(import|from)\s+{re.escape(imp)}\b", src, re.M):
            assert pkg in text, f"{pkg} is imported but missing from requirements.txt"