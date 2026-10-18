import re
from typing import Dict, Any, List

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

def extract_structured_context(transcript: str) -> dict:
    sentences = [s.strip() for s in re.split(r"[.\n]+", transcript) if s.strip()]
    
    decisions = []
    constraints = []
    failed_approaches = []
    next_steps = []
    
    for sentence in sentences:
        s_lower = sentence.lower()
        if any(w in s_lower for w in ["decided", "chose", "selected", "using"]):
            decisions.append(sentence)
        elif any(w in s_lower for w in ["do not", "must", "constraint", "keep"]):
            constraints.append(sentence)
        elif any(w in s_lower for w in ["failed", "error", "bug", "deprecated", "issue"]):
            failed_approaches.append(sentence)
        elif any(w in s_lower for w in ["next", "todo", "need to", "implement"]):
            next_steps.append(sentence)

    return {
        "tech_stack": ["FastAPI", "SQLite", "Python", "Gemma 2"],
        "decisions_made": decisions if decisions else ["Maintained existing backend layout"],
        "constraints": constraints if constraints else ["Do NOT break existing API response format"],
        "failed_approaches": failed_approaches if failed_approaches else ["None explicitly reported"],
        "next_step": next_steps[0] if next_steps else "Proceed with integration and testing"
    }

def format_handoff_markdown(project_name: str, data: dict = None) -> str:
    # Support both format_handoff_markdown(project_name, data) and format_handoff_markdown(data, project_name)
    if isinstance(project_name, dict) and isinstance(data, str):
        data, project_name = project_name, data
    data = data or {}
    return f"""# 🔄 ContextSwitch Handoff Artifact
**Project:** {project_name}

### 🎯 Confirmed Decisions
{chr(10).join(f"- {d}" for d in data.get('decisions_made', []))}

### ⚠️ Strict Constraints (DO NOT ALTER)
{chr(10).join(f"- {c}" for c in data.get('constraints', []))}

### ❌ Discarded / Failed Approaches
{chr(10).join(f"- {f}" for f in data.get('failed_approaches', []))}

### 🚀 Immediate Next Task
- {data.get('next_step', 'None')}
"""

# Aliases for handoff router
extract_context_items = extract_structured_context
format_markdown_artifact = format_handoff_markdown