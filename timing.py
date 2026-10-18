import time
from dotenv import load_dotenv; load_dotenv()
from app.services import compressor as C, protect as P
from app.services.gemma_entropy import count_tokens
text = open("big_prompt.txt", encoding="utf-8").read()
def t(name, fn):
    s = time.perf_counter(); r = fn(); print(f"{name:20} {time.perf_counter()-s:7.2f}s"); return r
t("mask", lambda: P.mask(text))
spans = t("protected_spans", lambda: P.protected_spans(text))
print("spans:", len(spans), " protected: %.0f%%" % (100 * sum(e - s for s, e in spans) / len(text)))
ruled = t("rule_strip", lambda: C._rule_strip(text))
t("missing_critical", lambda: P.missing_critical(text, ruled))
t("full compress", lambda: C.compress_prompt(text, "hybrid", 0.4))