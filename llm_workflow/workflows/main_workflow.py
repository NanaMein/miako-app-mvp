from typing import Optional, Any, Union
from crewai.flow import Flow, start, listen, router, or_
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from llm_workflow.vector_stores.vector_memory_store import ConversationMemoryStore
from llm_workflow.chat_completions.groq_llm import ChatCompletionsClass
from llm_workflow.prompts.prompt_library import PromptLibrary
from pathlib import Path
from fastapi import status, HTTPException
from crewai.flow.flow import FlowStreamingOutput


CURRENT_FILE_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_FILE_DIR.parent
PROMPT_DIR = PROJECT_ROOT / "prompts"
PROMPTS_PATH = PROMPT_DIR / "prompts.yaml"
LIBRARY = PromptLibrary(file_path=str(PROMPTS_PATH))
MEMORY_STORE = ConversationMemoryStore()


def date_time_now() -> str:
    return datetime.now(timezone.utc).isoformat()


from typing import Literal
class IntentResponse(BaseModel):
    reasoning: str
    confidence: float
    action: Literal["web_search", "rag_query", "direct_reply", "system_op"]
    parameters: dict

async def testing_intent_classifier(input_user: str):
    _system_prompt = LIBRARY.get_prompt("intent_classifier.gemini_series.version_1")
    _chatbot = ChatCompletionsClass()
    _chatbot.add_system(_system_prompt)
    _chatbot.add_user(input_user)
    _chat_resp = await _chatbot.groq_maverick()
    intent_data = IntentResponse.model_validate_json(_chat_resp)
    return intent_data



class MainFlowStates(BaseModel):
    input_message: str = Field(default="", description="User input message to llm workflow")
    input_user_id: str = Field(default="")
    async_session: Optional[AsyncSession] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    time_stamp: str = Field(default_factory=lambda: date_time_now())

class LLMWorkflow(Flow[MainFlowStates]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @start()
    async def is_it_english(self):
        system_prompt = LIBRARY.get_prompt("language_classifier.gemini_series.version_1")
        chatbot = ChatCompletionsClass()
        chatbot.add_system(system_prompt)
        chatbot.add_user(self.state.input_message)
        chat_response = await chatbot.groq_scout(
            max_completion_tokens=1,
        )
        return chat_response

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
        chatbot = ChatCompletionsClass()
        chatbot.add_system(system_prompt)
        chatbot.add_user(self.state.input_message)

        chat_response = await chatbot.groq_maverick()
        return chat_response


    @listen("ROUTER_DENIED")
    def unknown_category(self):
        print("UNKNOWN")
        return "violence and war"

    @listen(or_(english_user_query, translating_user_query, unknown_category))
    async def intent_classifier(self, answer):
        system_prompt = LIBRARY.get_prompt("intent_classifier.gemini_series.version_1")

        chatbot = ChatCompletionsClass()
        chatbot.add_system(system_prompt)
        chatbot.add_user(answer)

        chat_response = await chatbot.groq_maverick()
        print(f"Intents: \n**{chat_response}**\n")
        print(f"Translations: \n**{answer}**\n")
        return chat_response



main_llm_workflow = LLMWorkflow()

async def flow_kickoff(input_user_id: str, input_message: str, async_session: Optional[AsyncSession] = None):
    """Deprecated: Please use the new async wrapper because this one cause race conditions,
    which is not good for an ideal concurrent ready llm workflow."""
    inputs = {
        "input_user_id": input_user_id,
        "input_message": input_message,
        "async_session": async_session
    }
    return await main_llm_workflow.kickoff_async(inputs=inputs) #or (**inputs)

async def llm_workflow_kickoff(
        input_user_id: str,
        input_message: str,
        async_session: Optional[AsyncSession] = None
) -> Union[FlowStreamingOutput, Any]:
    inputs: dict = {
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