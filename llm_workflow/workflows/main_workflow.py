from typing import Optional, Any, Union
from crewai.flow import Flow, start, listen, router, or_
from crewai.types.streaming import FlowStreamingOutput
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage
from llm_workflow.llm.groq_llm import ChatCompletionsClass
from llm_workflow.prompts.prompt_library import PromptLibrary
from fastapi import status, HTTPException
from typing import Literal, Protocol
from dataclasses import dataclass, asdict


class AppResources:
    # current_file_dir = Path(__file__).parent
    # project_root = current_file_dir.parent
    # prompt_dir = project_root / "prompts"
    # prompts_path = prompt_dir / "prompts.yaml"
    # library = PromptLibrary(file_path=str(prompts_path))
    library = PromptLibrary()

RESOURCES = AppResources()


def date_time_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IntentResponse(BaseModel):
    reasoning: str
    confidence: float
    action: Literal["web_search", "rag_query", "direct_reply", "system_op"]
    parameters: dict



class MainFlowStates(BaseModel):
    input_message: str = Field(default="", description="User input message to llm workflow")
    input_user_id: str = Field(default="")
    intent_data: Optional[IntentResponse] = Field(default=None, description="Current intent data after translation")
    async_session: Optional[AsyncSession] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    time_stamp: str = Field(default_factory=lambda: date_time_now())

class AdaptiveConversationEngine(Flow[MainFlowStates]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.language_classifier_llm = ChatCompletionsClass()
        self.translation_llm = ChatCompletionsClass()
        self.intent_classifier_llm = ChatCompletionsClass()


    @property
    def message_storage(self):
        return MessageStorage(self.state.input_user_id)


    @start()
    async def safety_content_moderator(self):
        pass #For development only, assume there is a content moderator here.


    @listen(safety_content_moderator)
    async def is_it_english(self):
        system_message = RESOURCES.library.get_prompt("language_classifier.gemini_series.version_1")
        self.language_classifier_llm.add_system(content=system_message)
        self.language_classifier_llm.add_user(content=self.state.input_message)
        return await self.language_classifier_llm.groq_scout(max_completion_tokens=1)

    @router(is_it_english)
    def english_router(self, answer):
        if not answer:
            return "Unknown"

        is_english = str(answer).strip().upper()

        if is_english == "YES":
            return "ROUTER_PASS"

        elif is_english == "NO":
            return "ROUTER_TRANSLATE"

        elif is_english == "UNKNOWN":
            return "ROUTER_DENIED"
        else:
            return "ROUTER_DENIED"

    @listen("ROUTER_PASS")
    def english_user_query(self):
        print("PASS")
        return self.state.input_message

    @listen("ROUTER_TRANSLATE")
    async def translating_user_query(self):
        print("TRANSLATE")
        system_prompt = RESOURCES.library.get_prompt("translation_layer.qwen_series.version_1")
        self.translation_llm.add_system(system_prompt)
        self.translation_llm.add_user(self.state.input_message)
        return await self.translation_llm.groq_maverick()


    @listen("ROUTER_DENIED")
    def unknown_category(self):
        print("UNKNOWN")
        return "violence and war"

    @listen(or_(english_user_query, translating_user_query, unknown_category))
    async def intent_classifier(self, answer):
        system_prompt = RESOURCES.library.get_prompt("intent_classifier.gemini_series.version_1")
        self.intent_classifier_llm.add_system(system_prompt)
        self.intent_classifier_llm.add_user(answer)
        chat_response = await self.intent_classifier_llm.groq_maverick()
        intent_data = IntentResponse.model_validate_json(chat_response)
        self.state.intent_data = intent_data
        return intent_data

    @router(intent_classifier)
    def intent_router(self, _intent_data):
        intent_object: IntentResponse = _intent_data
        intent_action = intent_object.action
        if intent_action == "web_search":
            return "WEB_SEARCH"
        elif intent_action == "rag_query":
            return "RAG_QUERY"
        elif intent_action == "direct_reply":
            return "DIRECT_REPLY"
        elif intent_action == "system_op":
            return "SYSTEM_OP"
        else:
            return "UNKNOWN"

    @listen("WEB_SEARCH")
    def web_search_route(self):
        return "web search SUCCESS"

    @listen("DIRECT_REPLY")
    def direct_reply_route(self):
        return "direct reply SUCCESS"

    @listen("RAG_QUERY")
    def rag_query_route(self):
        return "rag query SUCCESS"

    @listen("SYSTEM_OP")
    def system_op_route(self):
        return self.state["raw_intent"]




@dataclass
class InputData:
    input_message: str
    input_user_id: str


class ChatEngineProtocol(Protocol):

    user_id: Union[str, Any]
    input_message: str

    @property
    def _input_data(self) -> dict[str, Any]: ...

    @property
    def flow_engine(self) -> Optional[Flow[MainFlowStates]]: ...

    async def run(self) -> Union[FlowStreamingOutput, str, None]: ...


class AdaptiveChatbot:
    def __init__(self, user_id: Union[str, Any], input_message: str):
        self.user_id = user_id
        self.input_message = input_message
        self._engine: Optional[AdaptiveConversationEngine] = None


    @property
    def flow_engine(self) -> Optional[Flow[MainFlowStates]]:
        if self._engine is None:
            self._engine = AdaptiveConversationEngine()
        return self._engine

    @property
    def _input_data(self) -> dict[str, Any]:
        inputs = InputData(input_user_id=self.user_id, input_message=self.input_message)
        return asdict(inputs)

    async def run(self) -> Union[FlowStreamingOutput, str, None]:
        try:
            response = await self.flow_engine.kickoff_async(inputs=self._input_data)
            if response is None:
                raise Exception("No output")
            return response
        except Exception as e:
            raise e


class ChatbotExecutor:
    def __init__(self, chat: ChatEngineProtocol):
        self.chat = chat

    async def execute(self):
        return await self.chat.run()


