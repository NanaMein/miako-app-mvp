from typing import Any, Union, Protocol
from pydantic import BaseModel
from crewai.flow.flow import Flow, FlowStreamingOutput




class ChatEngineProtocol(Protocol):

    user_id: Union[str, Any]
    input_message: str

    @property
    def _input_data(self) -> dict[str, Any]: ...

    @property
    def flow_engine(self) -> Flow[BaseModel]: ...

    async def run(self) -> Union[FlowStreamingOutput, str, None]: ...