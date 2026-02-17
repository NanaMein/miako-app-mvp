from typing import Union, Any, Literal, Optional
from crewai.flow.flow import Flow, start, listen, and_, or_
from pydantic import BaseModel, ConfigDict
from llm_workflow.llm.groq_llm import GroqLLM, MODEL
from llm_workflow.prompts.prompt_library import BasePrompt
from groq.types.chat import ChatCompletionMessage
import asyncio


class IntentLibrary(BasePrompt):
    def __init__(self):
        super().__init__("intent.yaml")

class AppResources:
    intent_library = IntentLibrary()
    llm = GroqLLM()

RESOURCES = AppResources()


class IntentResponse(BaseModel):
    reasoning: str
    confidence: float
    action: Literal["web_search", "rag_query", "direct_reply", "system_op"]
    parameters: dict


class IntentState(BaseModel):
    user_id: Union[str, Any] = ""
    input_message: str = ""
    unparsed_intent_data: str = ""
    intent_data: Optional[IntentResponse] = None
    system_prompt: str = RESOURCES.intent_library.get_prompt("intent_classifier.current")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class IntentClassifier(Flow[IntentState]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.llm = GroqLLM()

    @start()
    async def intent_classifier(self):
        self.llm.add_system(self.state.system_prompt)
        self.llm.add_user(self.state.input_message)
        response = await self.llm.groq_message_object(model=MODEL.maverick)
        return response


    @listen(intent_classifier)
    def data_parsing(self, initial_intent_data: ChatCompletionMessage | str):
        if isinstance(initial_intent_data, ChatCompletionMessage):
            intent_string = initial_intent_data.content
        else:
            intent_string = initial_intent_data

        intent_json = self._intent_validate_json(intent_string)
        return intent_json

    @listen(data_parsing)
    def intent_action(self, intent_json: IntentResponse):
        action = self._intent_action_router(intent_data=intent_json)
        return action


    @staticmethod
    def _intent_action_router(intent_data: IntentResponse):
        intent_action = intent_data.action
        if intent_action == "web_search":
            return "WEB_SEARCH"
        elif intent_action == "rag_query":
            return "RAG_QUERY"
        elif intent_action == "direct_reply":
            return "DIRECT_REPLY"
        elif intent_action == "system_op":
            return "SYSTEM_OP"
        else:
            return "UNKNOWN"


    @staticmethod
    def _intent_validate_json(data: str):
        _intent_json = IntentResponse.model_validate_json(data)
        return _intent_json


x = IntentClassifier()
_xx = x.kickoff_async({"input_message":"hello world"})
xxx = asyncio.run(_xx)
print(xxx)
