from typing import Literal, Any, Union, Optional
from crewai.flow.flow import Flow, start, listen, router, and_, or_
from pydantic import BaseModel, ConfigDict
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage
from llm_workflow.prompts.prompt_library import PromptLibrary
from llm_workflow.llm.groq_llm import ChatCompletionsClass as LLMGroq


library = PromptLibrary()



async def was_it_english(input_message: str):
    llm = LLMGroq()
    system_message = library.get_prompt("language_classifier.gemini_series.version_1")
    llm.add_system(content=system_message)
    llm.add_user(content=input_message)
    return await llm.groq_scout(max_completion_tokens=1)


def language_router(answer:str):
    if answer is None or answer == "":
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

async def storing_memory(orig_memory: MessageStorage, trans_memory: MessageStorage, message_to_be_saved:str):
    await orig_memory.add_human_message(message_to_be_saved)
    await trans_memory.add_human_message(message_to_be_saved)


class LanguageState(BaseModel):
    user_id: str = ""
    original_message: str = ""
    model_config = ConfigDict(arbitrary_types_allowed=True)





class _LanguageRouter(Flow[LanguageState]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)


    @start
    def english_identifier(self) -> str:



