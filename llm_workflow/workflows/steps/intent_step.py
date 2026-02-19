from typing import Union, Any, Literal, Optional
from crewai.flow.flow import Flow, start, listen, and_, or_
from pydantic import BaseModel, ConfigDict
from llm_workflow.llm.groq_llm import GroqLLM, MODEL
from llm_workflow.prompts.prompt_library import IntentLibrary, DataExtractorLibrary
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage
from groq.types.chat import ChatCompletionMessage
from llama_index.core.prompts import PromptTemplate
import asyncio




class AppResources:
    _intent_library = IntentLibrary()
    intent_classifier_instructions = _intent_library.get_prompt("intent_classifier.current")
    _test_intent_extractor = DataExtractorLibrary()
    _data_extractor_template = _test_intent_extractor.get_prompt("version-2")
    data_extractor = PromptTemplate(_data_extractor_template)

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

    model_config = ConfigDict(arbitrary_types_allowed=True)


class IntentClassifier(Flow[IntentState]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.llm = GroqLLM()

    @start()
    def start_with_data_extraction(self):
        pass

    # @start()
    # async def intent_classifier(self):
    #     self.llm.add_system(RESOURCES.intent_classifier_instructions)
    #     self.llm.add_user(self.state.input_message)
    #     response = await self.llm.groq_message_object(model=MODEL.maverick)
    #     return response



    # @listen(intent_classifier)
    # def data_parsing(self, initial_intent_data: ChatCompletionMessage | str):
    #     if isinstance(initial_intent_data, ChatCompletionMessage):
    #         intent_string = initial_intent_data.content
    #     else:
    #         intent_string = initial_intent_data
    #
    #     intent_json = self._intent_validate_json(intent_string)
    #     return intent_json
    #
    # @listen(data_parsing)
    # def intent_action(self, intent_json: IntentResponse):
    #     action = self._intent_action_router(intent_data=intent_json)
    #     return action


    @property
    def original_memory(self):
        _user_id = f"original_x_{self.state.user_id}"
        return MessageStorage(user_id=_user_id)

    @property
    def translated_memory(self):
        _user_id = f"translated_x_{self.state.user_id}"
        return MessageStorage(user_id=_user_id)



    # @staticmethod
    # def _intent_action_router(intent_data: IntentResponse):
    #     intent_action = intent_data.action
    #     if intent_action == "web_search":
    #         return "WEB_SEARCH"
    #     elif intent_action == "rag_query":
    #         return "RAG_QUERY"
    #     elif intent_action == "direct_reply":
    #         return "DIRECT_REPLY"
    #     elif intent_action == "system_op":
    #         return "SYSTEM_OP"
    #     else:
    #         return "UNKNOWN"
    #
    #
    # @staticmethod
    # def _intent_validate_json(data: str):
    #     _intent_json = IntentResponse.model_validate_json(data)
    #     return _intent_json

    @staticmethod
    def memory_parsing_to_string(input_list: list[Any]) -> str:
        _list = []
        for msg in input_list:
            role = msg["role"].upper()
            content = msg["content"]
            metadata = msg["metadata"]
            msg_str = f"{role}:\n{content}\nMetadata:\n{metadata}\n===\n"
            _list.append(msg_str)
        full_str = "".join(_list)
        return full_str

    async def _first_phase_intent_extractor(self):
        _orig_list = await self.original_memory.get_messages(include_metadata=True)
        _tran_list = await self.translated_memory.get_messages(include_metadata=True)
        original_str = self.memory_parsing_to_string(_orig_list)
        translated_str = self.memory_parsing_to_string(_tran_list)
        full_conversation_history = f"""
        === ORIGINAL MESSAGES ===\n
        {original_str}\n\n
        === TRANSLATED MESSAGES ===\n
        {translated_str}
        
        """


        formatted_prompt = RESOURCES.data_extractor.format(
            user_input=self.state.input_message,
            conversation_history=full_conversation_history,
            documentation_context="None",
        )
        return formatted_prompt


x = IntentClassifier()
_xx = x.kickoff_async({"input_message":"hello world"})
xxx = asyncio.run(_xx)
print(xxx)
