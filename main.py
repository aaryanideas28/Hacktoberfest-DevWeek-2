import os
from dotenv import load_dotenv

# 1. Load environment variables before any other imports
load_dotenv()
print("[startup] 1/5 .env loaded successfully.")
print(f"[startup] Config: GOOGLE_API_KEY={'set' if os.getenv('GOOGLE_API_KEY') else 'not set'} | "
      f"GEMINI_API_KEY={'set' if os.getenv('GEMINI_API_KEY') else 'not set'} | "
      f"OPENROUTER_API_KEY={'set' if os.getenv('OPENROUTER_API_KEY') else 'not set'} | "
      f"GEMMA_API_MODEL={os.getenv('GEMMA_API_MODEL', 'not set')}")

# 2. Framework imports
print("[startup] 2/5 Importing FastAPI & dependencies...")
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 3. Tracing initialization
print("[startup] 3/5 Initializing Sentry Tracing...")
from app.core.sentry import init_sentry
init_sentry()

# 4. Routers and DB initialization
print("[startup] 4/5 Initializing SQLite database...")
from app.db import init_db
init_db()

print("[startup] 5/5 Mounting routers...")
from app.routers import compress, handoff
from app.services.extractor import extract_items

app = FastAPI(title="ContextSwitch")

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