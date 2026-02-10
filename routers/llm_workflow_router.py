from typing import Union, Any

from fastapi import APIRouter, HTTPException, status
from llm_workflow.workflows.executor import ChatbotExecutor
from llm_workflow.workflows.flows import AdaptiveChatbot
from pydantic import BaseModel, Field



router = APIRouter(
    prefix="/api/chatbot",
    tags=["chatbot"],
)

class MessageResponse(BaseModel):
    message: str


class MessageRequest(MessageResponse):
    id: Union[str, Any] = Field(default="user_test")




@router.post("/send-message", response_model=MessageResponse)
async def send_message(request: MessageRequest):
    try:
        chat_obj = AdaptiveChatbot(
            user_id=request.id,
            input_message=request.message,
        )
        chatbot = ChatbotExecutor(chat_obj)
        response = await chatbot.execute()
        return MessageResponse(message=str(response))
    except HTTPException:
        raise
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(err))