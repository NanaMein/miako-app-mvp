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




class LanguageState(BaseModel):
    user_id: str = ""
    original_message: str = ""
    model_config = ConfigDict(arbitrary_types_allowed=True)





class _LanguageRouter(Flow[LanguageState]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.llm = LLMGroq()

    @property
    def original_memory(self):
        _user_id = f"original_x_{self.state.user_id}"
        return MessageStorage(user_id=_user_id)

    @property
    def translated_memory(self):
        _user_id = f"translated_x_{self.state.user_id}"
        return MessageStorage(user_id=_user_id)

    @start
    async def english_identifier(self) -> str:
        return await self._english_identifier(self.state.original_message)


    @router(english_identifier)
    def english_router(self, identified_language: str):
        return self._english_router(identified_language)

    @listen("english_router_passed")
    def english_router_passed(self):
        return self.state.original_message

    @listen("english_router_failed")
    async def english_router_failed(self):
        return await self._translate_to_english(self.state.original_message)

    @listen(or_(english_router_passed, english_router_failed))
    async def memory_update(self, message):
        await self.original_memory.add_human_message(self.state.original_message)
        await self.translated_memory.add_human_message(message)
        return message

    @listen(memory_update)
    def final_answer(self, message):
        return message




    async def _english_identifier(self, input_message: str):
        system_message = LIB.get_prompt("language_classifier.current")
        self.llm.add_system(system_message)
        self.llm.add_user(input_message)
        response = await self.llm.groq_scout(max_completion_tokens=1)
        return str(response)


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

    async def _translate_to_english(self, input_message: str):
        system_message = LIB.get_prompt("")
        self.llm.add_system(system_message)
        self.llm.add_user(input_message)
        response = await self.llm.groq_maverick()
        return str(response)




    @listen("error_db")
    def error_db(self):
        return None
