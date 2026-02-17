from typing import Optional, Union, Any, Protocol, Self, List, Dict
from groq import AsyncGroq
from groq.types.chat import (
    ChatCompletionMessage,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam
)
from functools import lru_cache
from llm_workflow.config_files.config import workflow_settings


class GroqModelList:
    compound_beta =  "compound-beta"
    compound_beta_mini = "compound-beta-mini"
    instant = "llama-3.1-8b-instant"
    versatile =  "llama-3.3-70b-versatile"
    scout = "meta-llama/llama-4-scout-17b-16e-instruct"
    maverick = "meta-llama/llama-4-maverick-17b-128e-instruct"
    llama_guard =  "meta-llama/llama-guard-4-12b"
    kimi =  "moonshotai/kimi-k2-instruct-0905"
    gpt_oss_20 = "openai/gpt-oss-20b"
    gpt_oss_120 = "openai/gpt-oss-120b"
    qwen = "qwen/qwen3-32b"

MODEL = GroqModelList()
ChatCompReturnType = List[
    ChatCompletionSystemMessageParam |
    ChatCompletionUserMessageParam |
    ChatCompletionAssistantMessageParam |
    Dict[str, Any]
]

@lru_cache()
def get_groq_client():
    return AsyncGroq(api_key=workflow_settings.GROQ_API_KEY.get_secret_value())

class ChatBase(Protocol):

    def add_system(self, content: str = "") -> Self:
        ...

    def add_user(self, content: str = "") -> Self:
        ...

    def add_assistant(self, content: str = "") -> Self:
        ...

    @property
    def cached_messages(self) -> ChatCompReturnType:
        ...




class ChatCompletionsClass:

    def __init__(self):
        self._cached_messages = None

    @property
    def cached_messages(self) -> ChatCompReturnType:
        if self._cached_messages is None:
            self._cached_messages = []
        return self._cached_messages

    @property
    def client(self) -> AsyncGroq:
        return get_groq_client()

    async def groq_scout(self, **kwargs):
        return await self._pipeline("scout",**kwargs)


    async def groq_maverick(self, **kwargs):
        return await self._pipeline("mave", **kwargs)


    async def groq_versatile(self, **kwargs):
        return await self._pipeline(model="vers", **kwargs)

    async def custom_groq(self, model_type: str, **kwargs):
        return await self._pipeline(model=model_type, **kwargs)


    async def _pipeline(self, model: str, **kwargs) -> str:
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("max_completion_tokens", 8000)
        kwargs.setdefault("top_p", 1)
        kwargs.setdefault("stop", None)
        kwargs.setdefault("stream", False)

        completion = await self.client.chat.completions.create(
            model=_model(model=model),
            messages=self.cached_messages,
            **kwargs
        )
        pre_content = completion.choices[0].message
        return pre_content.content


    def add_system(self, content: str = ""):
        self._add_msg("system", content)
        return self

    def add_user(self, content: str = ""):
        self._add_msg("user",content)
        return self

    def add_assistant(self, content: str = ""):
        self._add_msg("assistant", content)
        return self

    def _add_msg(self, role: str, content: str = ""):
        if content and content.strip():
            self.cached_messages.append({"role": role, "content": content})


def _model(model: str) -> Optional[str]:
    choices = {
        "cb": "compound-beta",
        "cbm": "compound-beta-mini",
        "inst": "llama-3.1-8b-instant",
        "vers": "llama-3.3-70b-versatile",
        "mave": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "scout": "meta-llama/llama-4-scout-17b-16e-instruct",
        "guard": "meta-llama/llama-guard-4-12b",
        "moon": "moonshotai/kimi-k2-instruct-0905",
        "oss120": "openai/gpt-oss-120b",
        "oss20": "openai/gpt-oss-20b",
        "qwen": "qwen/qwen3-32b"
    }

    return choices.get(model)


class GroqLLM(ChatBase):

    def __init__(self):
        super().__init__()

    @property
    def client(self) -> AsyncGroq:
        return get_groq_client()

    def add_system(self, content: str = ""):
        self._add_msg("system", content)
        return self

    def add_user(self, content: str = ""):
        self._add_msg("user",content)
        return self

    def add_assistant(self, content: str = ""):
        self._add_msg("assistant", content)
        return self

    def _add_msg(self, role: str, content: str = ""):
        if content and content.strip():
            self.cached_messages.append({"role": role, "content": content})


    @staticmethod
    def _setting_defaults(**kwargs):
        kwargs.setdefault("temperature", 0)
        kwargs.setdefault("max_completion_tokens", 8000)
        kwargs.setdefault("top_p", 1)
        kwargs.setdefault("stop", None)
        kwargs.setdefault("stream", False)
        return kwargs


    async def _pipeline(self, model: str, return_as_object: bool, **kwargs):
        _kwargs = self._setting_defaults(**kwargs)

        completion = await self.client.chat.completions.create(
            model=model,
            messages=self.cached_messages,
            **_kwargs
        )
        pre_content = completion.choices[0].message

        if return_as_object:
            return pre_content
        else:
            return pre_content.content

    async def groq_chat(self, model: str, **kwargs) -> str:
        return await self._pipeline(model=model, return_as_object=False, **kwargs)

    async def groq_message_object(self, model: str, return_as_object: bool = True, **kwargs) -> Union[str, ChatCompletionMessage]:
        return await self._pipeline(model=model, return_as_object=return_as_object, **kwargs)