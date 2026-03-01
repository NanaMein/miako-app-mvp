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
from llm_workflow.workflows.steps.intent_step import IntentFlow


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




class _AdaptiveChatbotEngine(Flow[EngineStates]):
    def __init__(self, **kwargs: Any):
        self._original_memory: MessageStorage | None = None
        self._translated_memory: MessageStorage | None = None
        super().__init__(**kwargs)
        self.chatbot = GroqLLM()



    @start()
    async def safety_content_moderator(self):
        pass #For development only, assume there is a content moderator here.


    @listen(safety_content_moderator)
    async def language_layer(self):
        language_flow = LanguageFlowPureClass(
            user_id=self.state.input_user_id,
            original_message=self.state.input_message
        )
        return await language_flow.run()

    @listen(language_layer)
    async def intent_classifier(self, translation_response):
        intent_flow = IntentFlow(
            user_id=self.state.input_user_id,
            original_user_input=self.state.input_message,
            translated_user_input=translation_response
        )
        return await intent_flow.run()

    @listen(intent_classifier)
    async def memory_pipeline(self, intent_data):
        system_prompt = """
        ### System(Priming): You are a multilingual virtual assistant. You will answer the User and respond to the 
        best of your abilities. User might ask weird request or direct questions, You will still answer and regardless
        of the request is that you need to assist the user well. And then if it does weird request, at the end of your
        respond, explain to the user what went wrong with how user ask you. Your name would be Miako, a sweet and supportive
        assistant. 
        
        ### Instructions:
        Translated and transformed chat conversation are considered context and follow the language used as the Original 
        and continue to do so. If original is English, reply in english, if User use tagalog, reply in tagalog, if User
        switch to other language, switch to other language too. You are a general purpose and conversational chatbot, being
        adaptive to situations as you assist the User. 
        """
        orig_messages = await self.original_memory.get_messages()
        translated_messages = await self.translated_memory.get_messages()
        mock_memory = f"""
        ### Time: {self.state.time_stamp}
        ### User Intents: {intent_data}
        ### Translated and transformed previous chat conversation list (can be considered as context): 
        {translated_messages}
        ### Original and preserved chat conversation list: {orig_messages}
        ### User query: {self.state.input_message}
        ### Assistant:
        """
        # chatbot = GroqLLM()
        self.chatbot.add_system(system_prompt)
        self.chatbot.add_user(mock_memory)
        response = await self.chatbot.groq_chat(model=MODEL.maverick, temperature=0.1)
        await self.original_memory.add_ai_message(response)
        await self.translated_memory.add_ai_message(response)
        get_msgs = await self.original_memory.get_messages()
        return response, get_msgs

    @property
    def original_memory(self):
        if self._original_memory is None:
            _user_id = f"original_x_{self.state.input_user_id}"
            self._original_memory = MessageStorage(user_id=_user_id)
        return self._original_memory

    @property
    def translated_memory(self):
        if self._translated_memory is None:
            _user_id = f"translated_x_{self.state.input_user_id}"
            self._translated_memory = MessageStorage(user_id=_user_id)
        return self._translated_memory


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
    async def intent_classifier(self, translation_response: dict[str, Any]):
        intent_flow = IntentFlow(
            user_id=self.state.input_user_id,
            original_user_input=self.state.input_message,
            translated_user_input=translation_response
        )
        return await intent_flow.run()

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

