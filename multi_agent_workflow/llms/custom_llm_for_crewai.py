import os
import logging
from enum import Enum
from typing import Any, List, Optional, Dict, Union
from groq import Groq
from crewai.llms.base_llm import BaseLLM
from crewai.utilities.types import LLMMessage
from crewai.agent.core import Agent
from crewai.task import Task

class GroqModelList(str, Enum):
    LLAMA_SCOUT = "meta-llama/llama-4-scout-17b-16e-instruct"
    LLAMA_MAVERICK = "meta-llama/llama-4-maverick-17b-128e-instruct"
    QWEN_QWEN = "qwen/qwen3-32b"
    VERSATILE = "llama-3.3-70b-versatile"
    INSTANT = "llama-3.1-8b-instant"
    OSS_20 = "openai/gpt-oss-20b"


class GroqLLM(BaseLLM):
    """
    models: scout, maverick, qwen, versatile, instant, oss-20b
    """

    MODEL_ALIASES = {
        "scout" : GroqModelList.LLAMA_SCOUT,
        "maverick" : GroqModelList.LLAMA_MAVERICK,
        "qwen" : GroqModelList.QWEN_QWEN,
        "versatile" : GroqModelList.VERSATILE,
        "instant" : GroqModelList.INSTANT,
        "oss-20b" : GroqModelList.OSS_20
    }

    def __init__(
            self,
            model: str = "scout",
            api_key: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            stop: Optional[List[str]] = None,
            **kwargs: Any,
    ) -> None:


        resolved_models = self.MODEL_ALIASES.get(model, model)

        if isinstance(resolved_models, Enum):
            resolved_models = resolved_models.value

        super().__init__(
            model=resolved_models,
            temperature=temperature,
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            **kwargs,
        )


        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing. Please set it in your .env or pass it in.")

        self.client = Groq(api_key=self.api_key)
        self.max_tokens = max_tokens
        self.stop = stop

    def call(
            self,
            messages: Union[str, List[LLMMessage]],
            tools: Optional[List[Dict]] = None,
            callbacks: Optional[List[Any]] = None,
            available_functions: Optional[Dict] = None,
            from_task: Optional[Task] = None,
            from_agent: Optional[Agent] = None,
            response_model: Optional[Any] = None,
            **kwargs: Any,
    ) -> str:


        try:

            formatted_messages = self._format_messages(messages)


            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": self.temperature,
            }

            if self.max_tokens:
                params["max_tokens"] = self.max_tokens
            if self.stop:
                params["stop"] = self.stop


            if response_model:
                params["response_format"] = {"type": "json_object"}



            if tools:
                params["tools"] = self._convert_tools_to_groq_format(tools)
                params["tool_choice"] = "auto"

            if self.model == GroqModelList.QWEN_QWEN.value:
                params["reasoning_effort"] = "default"
                params["reasoning_format"] = "hidden"
                params["top_p"] = 0.95

            if self.model == GroqModelList.OSS_20.value:
                params["tools"] = [{"type":"browser_search"}]
                params["reasoning_effort"] = "medium"
                params["max_tokens"] = 60000

            response = self.client.chat.completions.create(**params)
            message = response.choices[0].message
            content = message.content or ""

            return content

        except Exception as e:
            logging.error(f"Groq API Error: {e}")
            raise e

    def _format_messages(self, messages: Union[str, List[LLMMessage]]) -> List[Dict[str, str]]:

        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]

        return [dict(m) for m in messages]

    def _convert_tools_to_groq_format(self, tools: List[Any]) -> List[Dict]:

        from crewai.llms.providers.utils.common import safe_tool_conversion

        openai_tools = []
        for tool in tools:
            name, description, parameters = safe_tool_conversion(tool, "OpenAI")
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            })
        return openai_tools

    def supports_function_calling(self) -> bool:
        return True