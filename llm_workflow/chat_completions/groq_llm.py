import asyncio
from typing import Optional
from groq import AsyncGroq
# from groq.types.chat import (
#     ChatCompletionUserMessageParam as ChatUser,
#     ChatCompletionSystemMessageParam as ChatSystem,
#     ChatCompletionAssistantMessageParam as ChatAssistant,
# )
from functools import lru_cache
from llm_workflow.config_files.config import settings_for_workflow as settings

@lru_cache()
def get_groq_client():
    return AsyncGroq(api_key=settings.GROQ_API_KEY.get_secret_value())


class ChatCompletionsClass_ver1:
    def __init__(self):
        self.cached_messages = []

    @property
    def client(self):
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
        print(f"""Here are the of caching \n\n {self.cached_messages}""")
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



bot = ChatCompletionsClass_ver1()
bot.add_system("YOu say the opposite of what is said to you by user")
bot.add_user("Hello")
bot.add_assistant("Goodbye!")
bot.add_user("please dont write -100 words")



bot_result = asyncio.run(bot.groq_scout(temperature=1,max_completion_tokens=10))
print(bot_result)
print("END")