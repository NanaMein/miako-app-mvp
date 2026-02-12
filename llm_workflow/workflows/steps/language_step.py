from typing import Literal, Any, Union, Optional
from crewai.flow.flow import Flow, start, listen, router, and_, or_
from pydantic import BaseModel, ConfigDict
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage
from llm_workflow.prompts.prompt_library import PromptLibrary, BasePrompt
from llm_workflow.llm.groq_llm import ChatCompletionsClass as LLMGroq


library = PromptLibrary()


class LanguageLibrary(BasePrompt):
    def __init__(self):
        super().__init__("language.yaml")

LIB = LanguageLibrary()


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
    async def english_identifier(self) -> str:
        return await self._english_identifier(self.state.original_message)


    @router(english_identifier)
    def english_router(self, identified_language: str):
        return self._english_router(identified_language)


    async def _english_identifier(self, input_message: str):
        system_message = LIB.get_prompt("language_classifier.current")
        self.llm.add_system(system_message)
        self.llm.add_user(input_message)
        response = await self.llm.groq_scout(max_completion_tokens=1)
        return response


    def _english_router(self, input_message: str):
        options = {"YES", "NO", "UNKNOWN"}
        upper_response = input_message.upper().strip()

        if upper_response in options:
            if upper_response == "YES":
                return "english_router_passed"
            elif upper_response == "NO":
                return "english_router_failed"
            elif upper_response == "UNKNOWN":
                return "error_db"
            else:
                return "error_db"

        return "error_db"




