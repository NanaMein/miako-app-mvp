from typing import Optional, Any
from crewai.flow import Flow, start, listen, router, or_
from llama_index.core import PromptTemplate
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv
from datetime import datetime, timezone
from groq.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from groq import AsyncGroq
from sqlmodel import select, desc
from models.message_model import Message
from schemas.message_schema import Role
from sqlalchemy.ext.asyncio import AsyncSession
from llm_workflow.vector_stores.vector_memory_store import ConversationMemoryStore
from llm_workflow.config_files.config import settings_for_workflow as settings


from llm_workflow.chat_completions.groq_llm import ChatCompletionsClass
from llm_workflow.prompts.prompt_library import PromptLibrary
from pathlib import Path
import os


CURRENT_FILE_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_FILE_DIR.parent
PROMPT_DIR = PROJECT_ROOT / "prompts"
PROMPTS_PATH = PROMPT_DIR / "prompts.yaml"
MEMORY_STORE = ConversationMemoryStore()
LIBRARY = PromptLibrary(file_path=str(PROMPTS_PATH))

def date_time_now() -> str:
    return datetime.now(timezone.utc).isoformat()

class FlowBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    time_stamp: str = Field(default_factory=lambda: date_time_now())


class MainFlowStates(FlowBase):
    input_message: str = Field(default="", description="User input message to llm workflow")
    input_user_id: str = Field(default="")
    async_session: Optional[AsyncSession] = None

class LLMWorkflow(Flow[MainFlowStates]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @start()
    async def is_it_english(self):
        system_prompt = LIBRARY.get_prompt("language_classifier.qwen_series.version_2")
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
        pass

    @listen("ROUTER_TRANSLATE")
    async def translating_user_query(self):
        system_prompt = LIBRARY.get_prompt("translation_layer.qwen_series.version_1")
        chatbot = ChatCompletionsClass()
        chatbot.add_system(system_prompt)
        chatbot.add_user(self.state.input_message)
        chat_response = await chatbot.groq_maverick()
        return chat_response


    @listen("ROUTER_DENIED")
    def unknown_category(self):
        pass

    @listen(or_(english_user_query, translating_user_query, unknown_category))
    async def intent_classifier(self, answer):
        system_prompt = LIBRARY.get_prompt("intent_classifier.qwen_series.version_1")
        chatbot = ChatCompletionsClass()
        chatbot.add_system(system_prompt)
        chatbot.add_user(answer)
        chat_response = await chatbot.groq_maverick()
        return chat_response



main_llm_workflow = LLMWorkflow()

async def flow_kickoff(input_user_id: str, input_message: str, async_session: Optional[AsyncSession] = None):
    inputs = {
        "input_user_id": input_user_id,
        "input_message": input_message,
        "async_session": async_session
    }
    return await main_llm_workflow.kickoff_async(inputs=inputs) #or (**inputs)



class MainWorkflowReference(Flow[MainFlowStates]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @property
    def chatbot(self):
        return ChatCompletionsClass()

    @property
    def memory_store(self):
        return MEMORY_STORE

    # @property
    # def prompt_lib(self):
    #     return PROMPT_LIBRARY

    async def add_memory_data(self, **kwargs):
        return await self.memory_store.add_memory(**kwargs)

    async def get_memory_data(self, **kwargs):
        return await self.memory_store.get_memory(**kwargs)

    @start()
    def start_with_load(self):
        chat = self.chatbot

    @listen(start_with_load)
    def ending(self):
        return "Success"
