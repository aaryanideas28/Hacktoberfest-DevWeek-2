import pytest
from app.services.compressor import compress_prompt_rule_based, compress_prompt, estimate_tokens
from app.services.gemma_entropy import compress_via_gemma_entropy

def test_estimate_tokens():
    assert estimate_tokens("hello world") >= 1
    assert estimate_tokens("") == 0

def test_compress_rule_based_strips_filler():
    prompt = "Can you please help me to write a function? Thank you!"
    result = compress_prompt_rule_based(prompt)
    assert "please" not in result["compressed"].lower()
    assert "thank you" not in result["compressed"].lower()
    assert result["tokens_saved"] >= 0

def test_compress_rule_based_protects_code_blocks():
    prompt = "Please run this code:\n```python\nprint('hello please')\n```\nThank you"
    result = compress_prompt_rule_based(prompt)
    assert "print('hello please')" in result["compressed"]

def test_gemma_entropy_pruning():
    prompt = "Can you please kindly help me to configure the database connection with parameters? Thank you very much."
    res = compress_via_gemma_entropy(prompt, drop_rate=0.40)
    assert res["compressed_prompt"] != ""
    assert res["tokens_saved"] >= 0
    assert "latency_ms" in res
    assert res["engine"] in ["gemma-2-2b-forward-pass", "information-theoretic-fallback", "protected-filler-fallback"]

def test_gemma_entropy_protects_code():
    prompt = "Please optimize this function:\n```python\ndef add(a, b):\n    return a + b\n```\nThanks a lot."
    res = compress_via_gemma_entropy(prompt, drop_rate=0.40)
    assert "def add(a, b):" in res["compressed_prompt"]

def test_unified_compress_modes():
    prompt = "Can you please explain FastAPI routers in detail? Thanks."
    res_hybrid = compress_prompt(prompt, mode="hybrid")
    assert res_hybrid["tokens_saved"] >= 0

    res_entropy = compress_prompt(prompt, mode="gemma_entropy")
    assert res_entropy["tokens_saved"] >= 0

def test_critical_tokens_protected():
    prompt = "Do not delete table users and keep port 8080 under all circumstances."
    res = compress_prompt(prompt, mode="hybrid")
    assert "not" in res["compressed_prompt"].lower()
    assert "8080" in res["compressed_prompt"]

