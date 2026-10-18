import re

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
        "tech_stack": ["FastAPI", "SQLite", "Python"],
        "decisions_made": decisions if decisions else ["Maintained existing backend layout"],
        "constraints": constraints if constraints else ["Do NOT break existing API response format"],
        "failed_approaches": failed_approaches if failed_approaches else ["None explicitly reported"],
        "next_step": next_steps[0] if next_steps else "Proceed with integration and testing"
    }

def format_handoff_markdown(project_name: str, data: dict) -> str:
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