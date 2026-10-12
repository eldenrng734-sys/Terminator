from pydantic import BaseModel


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class ErrorResponse(BaseModel):
    error: str