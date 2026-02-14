from typing import Any, Union, Protocol
from pydantic import BaseModel
from crewai.flow.flow import Flow
from fastapi import HTTPException, status




class ChatEngineProtocol(Protocol):

    user_id: Union[str, Any]
    input_message: str

    @property
    def _input_data(self) -> dict[str, Any]: ...

    @property
    def flow_engine(self) -> Flow[BaseModel]: ...

    async def run(self) -> Union[Any, str, None]: ...





class ChatbotExecutor:
    def __init__(self, chat: ChatEngineProtocol):
        self.chat = chat

    async def execute(self):
        try:
            return await self.chat.run()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))