from typing import Any
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage
from llm_workflow.prompts.prompt_library import LanguageLibrary
from llm_workflow.llm.groq_llm import MODEL, GroqLLM
from dataclasses import dataclass
from fastapi import HTTPException, status



LANGUAGE = LanguageLibrary()

@dataclass(slots=True)
class ValueStates:
    user_id: str
    original_message: str
    translated_message: str = ""

class LanguageFlowPureClass:
    def __init__(self, user_id: str, original_message: str):
        self.state = ValueStates(user_id=user_id, original_message=original_message)
        self.language = LANGUAGE

    @property
    def original_memory(self):
        _user_id = f"original_x_{self.state.user_id}"
        return MessageStorage(user_id=_user_id)

    @property
    def translated_memory(self):
        _user_id = f"translated_x_{self.state.user_id}"
        return MessageStorage(user_id=_user_id)


    @staticmethod
    async def groq_chat(system: str, model: str, input_message: str, max_completion_tokens: int, **kwargs) -> str:
        llm = GroqLLM()
        llm.add_system(content=system)
        llm.add_user(content=input_message)
        return await llm.groq_chat(model=model, max_completion_tokens=max_completion_tokens, **kwargs)

    async def _english_identifier(self):
        system_message = self.language.get_prompt("system-prompt.language-classifier")
        response = await self.groq_chat(
            system=system_message,
            input_message=self.state.original_message,
            model=MODEL.scout,
            temperature=.1,
            max_completion_tokens=1
        )
        return response

    async def _translate_to_english(self):
        system_message = self.language.get_prompt("system-prompt.language-translator")
        response = await self.groq_chat(
            system=system_message,
            input_message=self.state.original_message,
            model=MODEL.gpt_oss_20,
            max_completion_tokens=8000,
            reasoning_effort="medium",
            tools=[{"type": "browser_search"}]
        )
        self.state.translated_message = response
        return self

    async def _memory_update(self):
        await self.original_memory.add_human_message(self.state.original_message)
        await self.translated_memory.add_human_message(self.state.translated_message)
        return True


    async def _internal_workflow(self) -> str:
        original = self.state.original_message

        error = HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal Error")

        _lang_identity = await self._english_identifier()

        upper_response = _lang_identity.upper().strip()

        if upper_response in ["YES", "NO"]:
            if upper_response == "NO":
                await self._translate_to_english()
            else:
                self.state.translated_message = original

        else:
            raise error

        memory_updated = await self._memory_update()

        if memory_updated:
            return self.state.translated_message
        else:
            raise error


    async def run(self)-> str | Any:
        try:
            return await self._internal_workflow()
        except Exception as e:
            raise e





# _lang = LanguageFlow("user_123", "hello")
#
# async def get_lang():
#     result = await _lang.run()
#     print(result)
#     return result
# asyncio.run(get_lang())