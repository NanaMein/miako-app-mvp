import asyncio
import json
import uuid
from typing import Optional, Any, Union
from crewai.flow import Flow, start, listen, router, or_
from crewai.types.streaming import FlowStreamingOutput
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage, MessageStorageV1
from llm_workflow.llm.groq_llm import GroqLLM, MODEL
from llm_workflow.prompts.prompt_library import PromptLibrary
from fastapi import status, HTTPException
from typing import Literal
from dataclasses import dataclass, asdict
from llm_workflow.workflows.steps.language_step import LanguageFlow, LanguageFlowPureClass
from llm_workflow.workflows.steps.intent_step import IntentFlow, IntentFlowTemporary


class EngineStates(BaseModel):
    input_message: str = ""
    input_user_id: str = ""
    model_config = ConfigDict(arbitrary_types_allowed=True)


class _AdaptiveChatbotEngine(Flow[EngineStates]):
    def __init__(self, **kwargs: Any):
        self._original_memory: MessageStorage | None = None
        self._translated_memory: MessageStorage | None = None
        self._memory_storage: MessageStorageV1 | None = None
        super().__init__(**kwargs)
        self.chatbot = GroqLLM()



    @start()
    async def safety_content_moderator(self):
        pass #For development only, assume there is a content moderator here.


    @listen(safety_content_moderator)
    async def language_layer(self) -> dict[str, Any]:
        language_flow = LanguageFlow(
            user_id=self.state.input_user_id,
            original_message=self.state.input_message
        )
        return await language_flow.run()

    @listen(language_layer)
    async def intent_classifier(self, translation_response: dict[str, Any]) -> Exception | str:
        intent_flow = IntentFlow(
            user_id=self.state.input_user_id,
            input_data_obj=translation_response
        )
        return await intent_flow.run()

    @listen(intent_classifier)
    async def final_answer_test(self, data: Exception | str):
        if isinstance(data, Exception):
            return Exception(str(data))

        memory = await self.memory.get_messages(include_metadata=True)
        full_memory = json.dumps(memory)
        intents = data
        full_text = f"""===FULL CONVERSATION HISTORY===\n
        {full_memory}\n
        ===INTENTS===\n
        {intents}\n
        ===END===\n
        """
        return full_text

    @property
    def memory(self) -> MessageStorageV1:
        if self._memory is None:
            self._memory = MessageStorageV1(user_id=self.state.input_user_id)
        return self._memory



class AppResources:
    library = PromptLibrary()

RESOURCES = AppResources()


def date_time_now() -> str: return datetime.now(timezone.utc).isoformat()



@dataclass
class InputData:
    input_message: str
    input_user_id: str



class IntentResponse(BaseModel):
    reasoning: str
    confidence: float
    action: Literal["web_search", "rag_query", "direct_reply", "system_op"]
    parameters: dict



class EngineStates(BaseModel):
    input_message: str = Field(default="", description="User input message to llm workflow")
    input_user_id: str = Field(default="")
    intent_data: Optional[IntentResponse] = Field(default=None, description="Current intent data after translation")
    async_session: Optional[AsyncSession] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    time_stamp: str = Field(default_factory=lambda: date_time_now())





class _AdaptiveChatbotEngineV1ForRefactor(Flow[EngineStates]):
    def __init__(self, **kwargs: Any):
        self._original_memory: MessageStorage | None = None
        self._translated_memory: MessageStorage | None = None
        self._memory: MessageStorageV1 | None = None
        super().__init__(**kwargs)
        self.chatbot = GroqLLM()



    @start()
    async def safety_content_moderator(self):
        pass #For development only, assume there is a content moderator here.


    @listen(safety_content_moderator)
    async def language_layer(self) -> dict[str, Any]:
        language_flow = LanguageFlow(
            user_id=self.state.input_user_id,
            original_message=self.state.input_message
        )
        return await language_flow.run()

    @listen(language_layer)
    async def intent_classifier(self, translation_response: dict[str, Any]) -> Exception | str:
        intent_flow = IntentFlow(
            user_id=self.state.input_user_id,
            input_data_obj=translation_response
        )
        return await intent_flow.run()

    @listen(intent_classifier)
    async def final_answer_test(self, data: Exception | str):
        if isinstance(data, Exception):
            return Exception(str(data))

        memory = await self.memory.get_messages(include_metadata=True)
        full_memory = json.dumps(memory)
        intents = data
        full_text = f"""===FULL CONVERSATION HISTORY===\n
        {full_memory}\n
        ===INTENTS===\n
        {intents}\n
        ===END===\n
        """
        return full_text

    @property
    def memory(self) -> MessageStorageV1:
        if self._memory is None:
            self._memory = MessageStorageV1(user_id=self.state.input_user_id)
        return self._memory

class AdaptiveChatbot:
    def __init__(self, user_id: Union[str, Any], input_message: str):
        self.user_id = user_id
        self.input_message = input_message
        self._engine: Optional[Flow[BaseModel]] = None


    @property
    def flow_engine(self) -> Flow[BaseModel]:
        if self._engine is None:
            self._engine = _AdaptiveChatbotEngine()
        return self._engine

    @property
    def _input_data(self) -> dict[str, Any]:
        inputs = InputData(input_user_id=self.user_id, input_message=self.input_message)
        return asdict(inputs)

    async def run(self) -> Any | str | None:
        try:
            response, _ = await self.flow_engine.kickoff_async(inputs=self._input_data)
            if response is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bad Request")
            return response
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

