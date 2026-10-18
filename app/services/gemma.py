import os
import json
from typing import Optional, Dict, Any

# Model identifier for Google Gemma 2 on Google AI Studio
GEMMA_MODEL_NAME = "models/gemma-2-9b-it"

def is_gemma_available() -> bool:
    """Check if the Gemma API key is configured."""
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

def get_gemma_client():
    """
    Configure and return generativeai client if API key is present.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai.GenerativeModel(GEMMA_MODEL_NAME)
    except Exception as e:
        print(f"[Gemma] Error initializing model client: {e}")
        return None

def neural_compress_with_gemma(prompt: str) -> Optional[str]:
    """
    Compress a verbose prompt into dense instructions using Gemma 2.
    Returns None if Gemma is unavailable, allowing callers to fall back to rules.
    """
    model = get_gemma_client()
    if not model:
        return None

    system_instruction = (
        "You are an expert prompt compressor. Your job is to rewrite the user's prompt "
        "into the densest, most token-efficient version possible without losing any instructions, "
        "constraints, code blocks, or variable names. Strip all conversational filler."
    )

    try:
        response = model.generate_content(f"{system_instruction}\n\nPROMPT TO COMPRESS:\n{prompt}")
        return response.text.strip() if response and response.text else None
    except Exception as e:
        print(f"[Gemma] Neural compression failed: {e}")
        return None

def neural_extract_with_gemma(transcript: str) -> Optional[Dict[str, Any]]:
    """
    Extract structured context (decisions, constraints, next steps) using Gemma 2.
    Returns None if Gemma is unavailable.
    """
    model = get_gemma_client()
    if not model:
        return None

    extract_prompt = f"""
    Analyze the following developer conversation/transcript. Extract key engineering context.
    Return ONLY a valid JSON object with the following keys:
    {{
        "tech_stack": ["list of technologies mentioned"],
        "decisions_made": ["list of firm engineering decisions"],
        "constraints": ["list of strict constraints or rules"],
        "failed_approaches": ["list of errors, bugs, or approaches that failed"],
        "next_step": "single immediate next task to do"
    }}

    CONVERSATION:
    {transcript}
    """

    try:
        response = model.generate_content(extract_prompt)
        text = response.text.strip()
        # Clean any markdown code fences if returned
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())
    except Exception as e:
        print(f"[Gemma] Neural extraction failed: {e}")
        return None
