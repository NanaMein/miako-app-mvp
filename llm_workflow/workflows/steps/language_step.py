from typing import Any, Literal
from crewai.flow.flow import Flow, start, listen, router, or_
from pydantic import BaseModel, ConfigDict
from llm_workflow.memory.short_term_memory.message_cache import MessageStorageV1
from llm_workflow.prompts.prompt_library import LanguageLibrary
from llm_workflow.llm.groq_llm import GroqLLM, MODEL
from fastapi import status, HTTPException
from dataclasses import dataclass
from datetime import datetime, timezone
from jinja2 import Template
import asyncio
import json
import re





class Language:
    _lang_lib = LanguageLibrary()
    classifier = _lang_lib.get_prompt("system-prompt.language-classifier")
    translator = _lang_lib.get_prompt("system-prompt.language-translator")
    _user_prompt_translator_from_lib = _lang_lib.get_prompt("user-prompt.language-translator")
    _user_prompt_template = Template(_user_prompt_translator_from_lib, enable_async=True)

    async def user_prompt_translator(self, current_input: str, conversation_history: list[dict[str, Any]] | str) -> str:
        if isinstance(conversation_history, str):
            _conversation_history = conversation_history
        else:
            _conversation_history = json.dumps(conversation_history, ensure_ascii=False)

        user_prompt = await self._user_prompt_template.render_async(
            current_input=current_input,
            conversation_history=_conversation_history
        )
        return user_prompt


LANGUAGE = Language()


class LanguageState(BaseModel):
    user_id: str = ""
    original_message: str = ""
    source_language: str = ""
    dict_answer: dict[str, Any] = {}
    most_recent_error: Exception | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)



class _LanguageRouter(Flow[LanguageState]):
    def __init__(self, **kwargs: Any):
        self._message_memory: MessageStorageV1 | None = None
        super().__init__(**kwargs)
        self.chat_identifier = GroqLLM()
        self.chat_translator = GroqLLM()


    @start()
    async def language_identifier(self) -> bool:
        print("Running: english_identifier")
        chat_response = await self._language_classifier_chat(self.state.original_message)

        is_valid_language = self._language_type_validation(chat_response)

        if is_valid_language:
            self.state.source_language = chat_response.lower()
            return True
        return False


    @router(language_identifier)
    def language_router(self, is_valid_language) -> Literal["ENGLISH_PASSED", "ENGLISH_FAILED", "error_db"]:
        print("Running: english_router")
        if is_valid_language:
            if self.state.source_language == "en":
                return "ENGLISH_PASSED"
            else:
                return "ENGLISH_FAILED"

        return "error_db"


    @listen("ENGLISH_PASSED")
    def english_router_passed(self):
        print("Running: english_router_passed")
        return self.state.original_message

    @listen("ENGLISH_FAILED")
    async def english_router_failed(self) -> str:
        print("Running: english_router_failed")
        return await self._translate_to_english(self.state.original_message)

    @listen("error_db")
    def error_function(self) -> bool:
        print("Running: error_db")
        return False

    @listen(or_(english_router_passed, english_router_failed))
    async def memory_update(self, processed_message: str) -> tuple[str, dict[str, Any]]:
        metadata = {
            "original_text": self.state.original_message,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_language": self.state.source_language,
        }
        await self.memory.add_human_message(
            content=processed_message,
            metadata=metadata
        )
        return processed_message, metadata


    @listen(memory_update)
    def preparing_final_function(self, data: tuple[str, dict[str, Any]]):
        print("Running: success_function")
        message, metadata = data
        translated_text = {"translated_text": message}
        full_text_dict = {**translated_text, **metadata}
        self.state.dict_answer = full_text_dict
        return True

    @listen(or_(preparing_final_function, error_function))
    def final_answer(self, is_success) -> dict[str, Any]:
        if is_success:
            return self.state.dict_answer
        raise Exception("Internal error")


    @property
    def memory(self):
        if self._message_memory is None:
            self._message_memory = MessageStorageV1(self.state.user_id)
        return self._message_memory

    async def _language_classifier_chat(self, input_message: str):
        self.chat_identifier.add_system(LANGUAGE.classifier)
        self.chat_identifier.add_user(input_message)
        response = await self.chat_identifier.groq_chat(
            model=MODEL.scout, temperature=.1, max_completion_tokens=1
        )
        return response


    @staticmethod
    def _language_type_validation(original_input_text: str):
        if not re.fullmatch(r"[a-z]{2}", original_input_text, re.IGNORECASE):
            return False
        return True

    async def _translate_to_english(self, input_message: str) -> str:
        conversation_history = await self.memory.get_messages(include_metadata=True)
        user_prompt = await LANGUAGE.user_prompt_translator(
            current_input=input_message,
            conversation_history=conversation_history
        )
        system_prompt = LANGUAGE.translator
        self.chat_translator.add_system(system_prompt)
        self.chat_translator.add_user(user_prompt)
        response = await self.chat_translator.groq_chat(
            model=MODEL.gpt_oss_20,
            max_completion_tokens=8000,
            reasoning_effort="medium",
            tools=[{"type": "browser_search"}]
        )
        return response



class LanguageFlow:
    def __init__(self, user_id: str, original_message: str):
        self.original_message = original_message
        self.user_id = user_id
        self.flow = _LanguageRouter()

    async def run(self) -> dict[str, Any]:
        #self.flow.plot()
        return await self.flow.kickoff_async(
            {
                "user_id": self.user_id,
                "original_message": self.original_message,
            }
        )


from llm_workflow.memory.short_term_memory.message_cache import MessageStorage
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
