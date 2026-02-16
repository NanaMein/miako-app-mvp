from typing import Union, Any, Literal, Optional
from crewai.flow.flow import Flow, start, listen, and_, or_
from pydantic import BaseModel, ConfigDict
from llm_workflow.llm.groq_llm import ChatCompletionsClass as LLMGroq
from llm_workflow.prompts.prompt_library import BasePrompt

class IntentLibrary(BasePrompt):
    def __init__(self):
        super().__init__("intent.yaml")

class AppResources:
    intent_library = IntentLibrary()
    llm = LLMGroq()

RESOURCES = AppResources()


class IntentResponse(BaseModel):
    reasoning: str
    confidence: float
    action: Literal["web_search", "rag_query", "direct_reply", "system_op"]
    parameters: dict


async def intent_classifier(input_message: str):
    system_prompt = RESOURCES.intent_library.get_prompt("intent_classifier.gemini_series.version_1")
    RESOURCES.llm.add_system(system_prompt)
    RESOURCES.llm.add_user(input_message)
    chat_response = await RESOURCES.llm.groq_maverick()
    return parsing_intent_data(input_intent=chat_response)

def parsing_intent_data(input_intent: str):
    intent_json_data = IntentResponse.model_validate_json(input_intent)
    return intent_json_data


def intent_router(_intent_data: IntentResponse):
    intent_action = _intent_data.action
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

class IntentState(BaseModel):
    user_id: Union[str, Any] = ""
    input_message: str = ""
    unparsed_intent_data: str = ""
    intent_data: Optional[IntentResponse] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class IntentClassifier(Flow[IntentState]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.llm = LLMGroq()

    @start()
    def gathering_data_prompts(self):
        RESOURCES.intent_library.get_prompt("intent_classifier.current")
