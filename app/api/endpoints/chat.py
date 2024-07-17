from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chat_service import get_chat_response

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Endpoint to receive a chat message and return a response.

    Args:
        request (ChatRequest): The request body containing the chat message.

    Returns:
        ChatResponse: The response body containing the chat response.
    """
    response = get_chat_response(request.message)
    return ChatResponse(response=response)
