from typing import Literal, Any, Union, Optional
from crewai.flow.flow import Flow, start, listen, router, and_, or_
from pydantic import BaseModel, ConfigDict
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage
from llm_workflow.prompts.prompt_library import LanguageLibrary
from llm_workflow.llm.groq_llm import ChatCompletionsClass as LLMGroq
import asyncio


LANGUAGE = LanguageLibrary()


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

    @start()
    async def english_identifier(self) -> str:
        print("Running: english_identifier")
        return await self._english_identifier(self.state.original_message)


    @router(english_identifier)
    def english_router(self, identified_language):
        print("Running: english_router")
        return self._english_router(identified_language)

    @listen("ENGLISH_PASSED")
    def english_router_passed(self):
        print("Running: english_router_passed")
        return self.state.original_message

    @listen("ENGLISH_FAILED")
    async def english_router_failed(self):
        print("Running: english_router_failed")
        return await self._translate_to_english(self.state.original_message)

    @listen(or_(english_router_passed, english_router_failed))
    async def memory_update(self, message):
        print(f"Running: memory_update -> {message}")
        await self.original_memory.add_human_message(self.state.original_message)
        await self.translated_memory.add_human_message(message)
        return message




    async def _english_identifier(self, input_message):
        system_message = LANGUAGE.get_prompt("language_classifier.current")
        self.llm.add_system(system_message)
        self.llm.add_user(input_message)
        response = await self.llm.groq_scout(max_completion_tokens=1)
        return str(response)


    def _english_router(self, input_message: str):
        options = {"YES", "NO", "UNKNOWN"}
        upper_response = input_message.upper().strip()

        if upper_response in options:
            if upper_response == "YES":
                return "ENGLISH_PASSED"
            elif upper_response == "NO":
                return "ENGLISH_FAILED"
            elif upper_response == "UNKNOWN":
                return "error_db"
            else:
                return "error_db"

        return "error_db"

    async def _translate_to_english(self, input_message: str):
        system_message = LANGUAGE.get_prompt("language_translation.current")
        self.llm.add_system(system_message)
        self.llm.add_user(input_message)
        response = await self.llm.groq_maverick(max_completion_tokens=8000)
        print(response)
        return str(response)




    @listen("error_db")
    def error_db(self):
        print("error_db")
        return None



    @listen(memory_update)
    def final_answer(self, message):
        print("Running: final_answer")
        return message


class LanguageFlow:
    def __init__(self, user_id: str, original_message: str):
        self.original_message = original_message
        self.user_id = user_id
        self.flow = _LanguageRouter()

    async def run(self):
        return await self.flow.kickoff_async(
            {
                "user_id": self.user_id,
                "original_message": self.original_message,
            }
        )



# _lang = LanguageFlow("user_123", "hello")
#
# async def get_lang():
#     result = await _lang.run()
#     print(result)
#     return result
# asyncio.run(get_lang())