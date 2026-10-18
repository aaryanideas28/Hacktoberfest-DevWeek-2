import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.core.sentry import init_sentry
from app.routers import compress, handoff
from app.db import init_db

# Load .env variables (GEMINI_API_KEY, SENTRY_DSN)
load_dotenv()

# Initialize Sentry Agent Tracing
init_sentry()

app = FastAPI(title="ContextSwitch")

# Initialize SQLite database
init_db()

# Mount API routers
app.include_router(compress.router)
app.include_router(handoff.router)

# Mount static files for UI
static_dir = os.path.join(os.path.dirname(__file__), "app", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# System Health / Status endpoint for UI badges
@app.get("/api/v1/health")
def health_check():
    gemma_active = bool(os.getenv("GEMINI_API_KEY"))
    sentry_active = bool(os.getenv("SENTRY_DSN"))
    return {
        "status": "healthy",
        "gemma_active": gemma_active,
        "sentry_active": sentry_active,
    }


# Serve Web UI at root
@app.get("/")
def home():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ContextSwitch backend is running!"}


# Legacy route maintained for baseline tests
class ConversationRequest(BaseModel):
    conversation: str

@app.post("/extract-context")
def extract_context(request: ConversationRequest):
    from app.services.extractor import extract_items
    return {
        "project_context": {
            "raw_conversation": request.conversation,
            **extract_items(request.conversation),
        }
    }