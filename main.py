from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import get_settings
from app.schemas import ChatResponse
from app.services.llm import LLMError, build_user_message, get_chat_reply
from app.services.session import session_store

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "app" / "static" / "index.html"
APP_JS = BASE_DIR / "app" / "app.js"
STYLES_CSS = BASE_DIR / "app" / "static" / "styles.css"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(APP_JS)


@app.get("/styles.css")
def styles_css() -> FileResponse:
    return FileResponse(STYLES_CSS, media_type="text/css")


@app.get("/api/profile")
def profile() -> dict[str, str]:
    settings = get_settings()
    return {"name": settings.profile_name, "plan": settings.plan_name}


@app.get("/api/sessions")
def list_sessions(q: str = "") -> list[dict[str, str]]:
    return session_store.list_sessions(q)


@app.post("/api/sessions")
def create_session() -> dict[str, str]:
    session_id = session_store.new_session()
    session = session_store.get_session(session_id)
    assert session is not None
    return {key: session[key] for key in ("id", "title", "created_at", "updated_at")}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, object]:
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return session


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    if session_store.get_session(session_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    session_store.clear(session_id)
    return {"deleted": True}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(""),
    session_id: str | None = Form(None),
    images: list[UploadFile] | None = File(default=None),
) -> ChatResponse:
    image_parts: list[tuple[bytes, str]] = []
    for upload in images or []:
        data = await upload.read()
        image_parts.append((data, upload.content_type or "application/octet-stream"))

    try:
        user_message = build_user_message(message, image_parts)
    except LLMError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    sid = session_id or session_store.new_session()
    history = session_store.get_history(sid)
    try:
        reply = get_chat_reply([*history, user_message])
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    session_store.append(sid, user_message, reply)
    return ChatResponse(session_id=sid, reply=reply)


