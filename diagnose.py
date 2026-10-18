import re, time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from app.services import compressor, gemma_entropy as ge, protect

PROMPTS = {
    "short": "Can you please summarize the following report. Do NOT include any names. Thanks!",
    "medium": ("I would like you to write a function that parses 3 CSV files and returns a dict. "
               "Please make sure it does not crash on empty rows. Use \"utf-8\" encoding and keep "
               "the API stable. <data>Hello customer 4521, your order is late.</data> Thank you."),
    # add real ones, including a long (1-5k token) prompt
}

def ms(fn, *a, **k):
    t = time.perf_counter(); r = fn(*a, **k); return r, (time.perf_counter() - t) * 1000

def clear_cache(): ge._SCORE_CACHE.update(key=None, scores=None)

# 1. Which engine is really running? ------------------------------------------
print("\n== 1. engine state")
m, tok = ge.load_gemma_local_model()
print("loaded:", m is not None, "| device:", ge.get_device(), "| load_failed:", ge._LOAD_FAILED)
if m is not None:
    p = next(m.parameters()); print("dtype:", p.dtype, "| param device:", p.device)

for name, text in PROMPTS.items():
    print(f"\n######## {name} ({len(text.split())} words)")

    # 2. Stage timing ---------------------------------------------------------
    print("== 2. stage timing")
    ruled, t_rule = ms(compressor._rule_strip, text)
    clear_cache(); _, t_cold = ms(ge.compress_via_gemma_entropy, text, 0.4)
    _, t_hit = ms(ge.compress_via_gemma_entropy, text, 0.4)      # cache hit
    clear_cache(); _, t_warm = ms(ge.compress_via_gemma_entropy, text, 0.4)
    print(f"rule {t_rule:.1f}ms | entropy first {t_cold:.0f}ms | cache-hit {t_hit:.0f}ms | warm forward {t_warm:.0f}ms")

    # 3. Retries, forward passes, final engine --------------------------------
    print("== 3. retries / forward passes")
    n = {"fwd": 0, "once": 0}
    o_fwd, o_once = ge._token_surprisals, compressor._compress_once
    def fwd(*a, **k): n["fwd"] += 1; return o_fwd(*a, **k)
    def once(*a, **k): n["once"] += 1; return o_once(*a, **k)
    clear_cache()
    with patch.object(ge, "_token_surprisals", fwd), patch.object(compressor, "_compress_once", once):
        r = compressor.compress_prompt(text, "hybrid", 0.4)
    print(f"attempts={n['once']} forward_passes={n['fwd']} engine={r['engine']!r} "
          f"saved={r['compression_ratio']}% missing={r['critical_missing']}")

    # 4. Which stage loses critical tokens? -----------------------------------
    print("== 4. stage attribution")
    masked, store = protect.mask(text)
    print("mask/unmask roundtrip identical:", protect.unmask(masked, store) == text)
    print("rule stage lost   :", protect.missing_critical(text, ruled))
    ent = ge.compress_via_gemma_entropy(text, 0.4)["compressed_prompt"]
    print("entropy stage lost:", protect.missing_critical(text, ent))

    # 5. Protection correctness + how much is protected -----------------------
    print("== 5. protection")
    words, spans = ge._word_spans(text), protect.protected_spans(text)
    disjoint = all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    flags = ge._protected_flags(words, spans)
    brute = [any(s < we and e > ws for s, e in spans) for ws, we in words]
    print(f"spans sorted+disjoint={disjoint} | flags==brute_force={flags == brute} "
          f"| protected share={sum(brute)/len(brute):.0%}")

    # 6. Requested vs achieved compression ------------------------------------
    print("== 6. requested vs achieved")
    for rate in (0.2, 0.4, 0.6, 0.8):
        r = ge.compress_via_gemma_entropy(text, rate)
        wb, wa = len(text.split()), len(r["compressed_prompt"].split())
        print(f"rate={rate}: words dropped={1 - wa/wb:.0%} tokens saved={r['compression_ratio']}% "
              f"engine={r['engine']}")

    # 7. Punctuation / sentence boundaries ------------------------------------
    print("== 7. punctuation")
    P = lambda s: len(re.findall(r"[.!?:;]", s))
    r = ge.compress_via_gemma_entropy(text, 0.6)["compressed_prompt"]
    print(f"sentence punctuation before={P(text)} after={P(r)}")

    # 8. Token-count consistency ----------------------------------------------
    print("== 8. token counting")
    saved_tok = ge._GEMMA_TOKENIZER
    ge._GEMMA_TOKENIZER = None;  heur = ge.count_tokens(text)
    ge._GEMMA_TOKENIZER = saved_tok; real = ge.count_tokens(text)
    print(f"heuristic={heur} real={real} diff={abs(heur-real)/max(real,1):.0%}")

# 9. Window chunking quality (needs a long prompt) -----------------------------
print("\n== 9. window chunking")
if m is not None:
    long_text = " ".join(PROMPTS.values()) * 20
    ids = tok(long_text)["input_ids"]
    big = ge._token_surprisals(ids, m, tok, ge.get_device(), 4096)
    small = ge._token_surprisals(ids, m, tok, ge.get_device(), 64)
    pairs = [(a, b) for a, b in zip(big, small) if a != float("inf") and b != float("inf")]
    print(f"{len(ids)} tokens, mean |score diff| window 4096 vs 64: "
          f"{sum(abs(a-b) for a,b in pairs)/len(pairs):.2f} nats")

# 10. Concurrency ---------------------------------------------------------------
print("\n== 10. concurrency")
texts = list(PROMPTS.values()) * 2
_, t_seq = ms(lambda: [compressor.compress_prompt(t, "hybrid") for t in texts])
_, t_par = ms(lambda: list(ThreadPoolExecutor(4).map(lambda t: compressor.compress_prompt(t, "hybrid"), texts)))
print(f"sequential {t_seq:.0f}ms vs 4 threads {t_par:.0f}ms (parallel should NOT be much slower)")