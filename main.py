from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from app.auth import oauth, require_user
from app.config import get_settings
from app.schemas import ChatResponse
from app.services import auth_store
from app.services.llm import LLMError, build_user_message, get_chat_reply
from app.services.session import session_store

settings = get_settings()
app = FastAPI()
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    https_only=(settings.env == "production"),
    same_site="lax",
)

BASE_DIR = Path(__file__).resolve().parent
INDEX_HTML = BASE_DIR / "app" / "static" / "index.html"
APP_JS = BASE_DIR / "app" / "app.js"
STYLES_CSS = BASE_DIR / "app" / "static" / "styles.css"
EMBLEM_PNG = BASE_DIR / "app" / "static" / "terminator-emblem.png"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(APP_JS)


@app.get("/styles.css")
def styles_css() -> FileResponse:
    return FileResponse(STYLES_CSS, media_type="text/css")


@app.get("/terminator-emblem.png")
def terminator_emblem() -> FileResponse:
    return FileResponse(EMBLEM_PNG, media_type="image/png")


@app.get("/api/auth/{provider}/login")
async def login(provider: str, request: Request):
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=404, detail="Unsupported authentication provider.")
    redirect_uri = (
        f"{settings.frontend_host.rstrip('/')}/api/auth/{provider}/callback"
    )
    return await client.authorize_redirect(request, redirect_uri)


@app.get("/api/auth/{provider}/callback")
async def callback(provider: str, request: Request):
    client = oauth.create_client(provider)
    if client is None:
        raise HTTPException(status_code=404, detail="Unsupported authentication provider.")
    token = await client.authorize_access_token(request)

    if provider == "google":
        userinfo = token.get("userinfo") or await client.userinfo(token=token)
        provider_user_id = userinfo["sub"]
        email = userinfo.get("email")
        name = userinfo.get("name")
        avatar = userinfo.get("picture")
    else:
        profile = (await client.get("user", token=token)).json()
        provider_user_id = str(profile["id"])
        name = profile.get("name") or profile.get("login")
        avatar = profile.get("avatar_url")
        email = profile.get("email")
        if not email:
            emails = (await client.get("user/emails", token=token)).json()
            primary = next((item for item in emails if item.get("primary")), None)
            email = primary["email"] if primary else None

    user = auth_store.get_or_create_user(
        provider, provider_user_id, email, name, avatar
    )
    request.session["user_id"] = user["id"]
    return RedirectResponse(url="/")


@app.get("/api/auth/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/")


@app.get("/api/me")
def me(user_id: int = Depends(require_user)) -> dict[str, object]:
    user = auth_store.get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@app.get("/api/profile")
def profile(user_id: int = Depends(require_user)) -> dict[str, str]:
    settings = get_settings()
    return {"name": settings.profile_name, "plan": settings.plan_name}


@app.get("/api/sessions")
def list_sessions(
    q: str = "", user_id: int = Depends(require_user)
) -> list[dict[str, str]]:
    return session_store.list_sessions(q, user_id=user_id)


@app.post("/api/sessions")
def create_session(user_id: int = Depends(require_user)) -> dict[str, str]:
    session_id = session_store.new_session(user_id=user_id)
    session = session_store.get_session(session_id, user_id=user_id)
    assert session is not None
    return {key: session[key] for key in ("id", "title", "created_at", "updated_at")}


@app.get("/api/sessions/{session_id}")
def get_session(
    session_id: str, user_id: int = Depends(require_user)
) -> dict[str, object]:
    session = session_store.get_session(session_id, user_id=user_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return session


@app.delete("/api/sessions/{session_id}")
def delete_session(
    session_id: str, user_id: int = Depends(require_user)
) -> dict[str, bool]:
    if session_store.get_session(session_id, user_id=user_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    session_store.clear(session_id, user_id=user_id)
    return {"deleted": True}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    user_id: int = Depends(require_user),
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

    sid = session_id or session_store.new_session(user_id=user_id)
    if session_id is not None and session_store.get_session(sid, user_id=user_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    history = session_store.get_history(sid, user_id=user_id)
    try:
        reply = get_chat_reply([*history, user_message])
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    session_store.append(sid, user_message, reply, user_id=user_id)
    return ChatResponse(session_id=sid, reply=reply)


