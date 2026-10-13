from authlib.integrations.starlette_client import OAuth
from fastapi import HTTPException, Request

from app.config import get_settings

settings = get_settings()

oauth = OAuth()

oauth.register(
    name="google",
    client_id=settings.google_client_id,
    client_secret=settings.google_client_secret,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

oauth.register(
    name="github",
    client_id=settings.github_client_id,
    client_secret=settings.github_client_secret,
    access_token_url="https://github.com/login/oauth/access_token",
    authorize_url="https://github.com/login/oauth/authorize",
    api_base_url="https://api.github.com/",
    client_kwargs={"scope": "user:email"},
)

def require_user(request: Request) -> int:
    user_id = request.session.get("user_id")
    if not isinstance(user_id, int) or user_id < 1:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_id