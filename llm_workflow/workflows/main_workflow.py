from typing import Optional, Any, Union
from crewai.flow import Flow, start, listen, router, or_
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from llm_workflow.memory.long_term_memory.memory_store import ConversationMemoryStore
from llm_workflow.llm.groq_llm import ChatCompletionsClass
from llm_workflow.prompts.prompt_library import PromptLibrary
from pathlib import Path
from fastapi import status, HTTPException
from crewai.flow.flow import FlowStreamingOutput
from typing import Literal



CURRENT_FILE_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_FILE_DIR.parent
PROMPT_DIR = PROJECT_ROOT / "prompts"
PROMPTS_PATH = PROMPT_DIR / "prompts.yaml"
LIBRARY = PromptLibrary(file_path=str(PROMPTS_PATH))
MEMORY_STORE = ConversationMemoryStore()


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
    async_session: Optional[AsyncSession] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    time_stamp: str = Field(default_factory=lambda: date_time_now())

class LLMWorkflow(Flow[MainFlowStates]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.language_classifier_llm = ChatCompletionsClass()
        self.translation_llm = ChatCompletionsClass()
        self.intent_classifier_llm = ChatCompletionsClass()


    @property
    def language_classifier_llm(self) -> ChatCompletionsClass:
        return ChatCompletionsClass()




    @start()
    async def is_it_english(self):
        system_prompt = LIBRARY.get_prompt("language_classifier.gemini_series.version_1")
        self.language_classifier_llm.add_system(content=system_prompt)
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
        system_prompt = LIBRARY.get_prompt("translation_layer.qwen_series.version_1")
        self.translation_llm.add_system(system_prompt)
        self.translation_llm.add_user(self.state.input_message)
        return await self.translation_llm.groq_maverick()


    @listen("ROUTER_DENIED")
    def unknown_category(self):
        print("UNKNOWN")
        return "violence and war"

    @listen(or_(english_user_query, translating_user_query, unknown_category))
    async def intent_classifier(self, answer):
        system_prompt = LIBRARY.get_prompt("intent_classifier.gemini_series.version_1")
        self.intent_classifier_llm.add_system(system_prompt)
        self.intent_classifier_llm.add_user(answer)
        chat_response = await self.intent_classifier_llm.groq_maverick()
        intent_data = IntentResponse.model_validate_json(chat_response)
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
        return "system OP SUCCESS"


async def flow_kickoff(
        input_user_id: str,
        input_message: str,
        async_session: Optional[AsyncSession] = None
) -> Union[FlowStreamingOutput, Any]:

    inputs = {
        "input_user_id": input_user_id,
        "input_message": input_message,
        "async_session": async_session
    }
    try:
        _flow_kickoff = LLMWorkflow()
        flow_result: FlowStreamingOutput = await _flow_kickoff.kickoff_async(inputs=inputs)
        return flow_result
    except Exception as e:
        HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bad request: {e}")
