import pytest
from app.services.compressor import compress_prompt_rule_based, estimate_tokens

def test_estimate_tokens():
    assert estimate_tokens("hello world") >= 1

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
