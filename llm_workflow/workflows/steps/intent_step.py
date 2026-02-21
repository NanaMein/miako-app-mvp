from typing import Union, Any, List
from crewai.flow.flow import Flow, start, listen, and_, or_, router
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from llm_workflow.llm.groq_llm import GroqLLM, MODEL
from llm_workflow.prompts.prompt_library import DataExtractorLibrary
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage
from llm_workflow.memory.short_term_memory._fake_memory_testing import fake_memory
from groq.types.chat import ChatCompletionMessage
from jinja2 import Template
import asyncio
import json


class AppResources:
    _data_extractor_prompts = DataExtractorLibrary()
    _user_first_phase_template = _data_extractor_prompts.get_prompt("user-first-phase")

    system_first_phase = _data_extractor_prompts.get_prompt("system-first-phase")
    user_first_phase = Template(_user_first_phase_template)
    documentation_context = _data_extractor_prompts.get_prompt("documentation-context")

RESOURCES = AppResources()

class Fact(BaseModel):
    confidence: float = Field(..., ge=0.0, le=1.0)
    topic: str
    fact: str
    relevance_reason: str

class ExtractionResponse(BaseModel):
    facts: List[Fact]
    message: str | None = None


class IntentState(BaseModel):
    user_id: Union[str, Any] = ""
    translated_user_input: str = ""
    original_user_input: str = ""
    response_from_first_phase: str = ""
    latest_error_catch: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class IntentClassifier(Flow[IntentState]):
    def __init__(self, **kwargs: Any):
        self._translated_memory: MessageStorage | None = None
        self._original_memory: MessageStorage | None = None
        self.in_development_phase: bool = False
        self.in_error_mode: bool = False
        super().__init__(**kwargs)
        self.llm = GroqLLM()




    @start()
    async def start_or_testing_phase(self):
        if self.in_development_phase:
            print('TESTING PHASE')
            return await _prompts_for_first_phase_mock(
                original_memory=self.original_memory,
                translated_memory=self.translated_memory
            )
        else:
            print("PRODUCTION PHASE")
            return await self._prompts_for_first_phase()


    @listen(start_or_testing_phase)
    async def start_with_data_extraction(self, prompts: tuple[str, str]):
        system_prompt, user_prompt = prompts
        self.llm.add_system(system_prompt)
        self.llm.add_user(user_prompt)
        response = await self.llm.groq_message_object(model=MODEL.scout, return_as_object=True, temperature=.1)
        return response

    @listen(start_with_data_extraction)
    def data_parsing(self, _resp):
        if isinstance(_resp, ChatCompletionMessage):
            response = _resp.content
        else:
            response = _resp

        if self.in_error_mode:
            is_valid = False
            error = "Hello world Error testing"
        else:
            is_valid, error = self._validate_extraction_response(response)

        if is_valid:
            return response
        else:
            self.state.latest_error_catch = error
            return "error_fallback_logic"

    @router(data_parsing)
    def checking_data_validation(self, data: str):
        if data == "error_fallback_logic":
            return data
        elif data != "error_fallback_logic":
            self.state.response_from_first_phase = data
            return "no_error_fallback_logic"
        else:
            return data


    @listen("error_fallback_logic")
    def error_fallback_catcher(self):
        print("=== STARTING ERROR FALLBACK LOGIC ===")
        return self.state.latest_error_catch

    @listen("no_error_fallback_logic")
    def success_no_error(self):
        print("=== STARTING SUCCESS and no FALLBACK LOGIC ===")
        return self.state.response_from_first_phase




    @property
    def original_memory(self):
        if self._original_memory is None:
            _user_id = f"original_x_{self.state.user_id}"
            self._original_memory = MessageStorage(user_id=_user_id)
        return self._original_memory

    @property
    def translated_memory(self):
        if self._translated_memory is None:
            _user_id = f"translated_x_{self.state.user_id}"
            self._translated_memory = MessageStorage(user_id=_user_id)
        return self._translated_memory



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

    def _validate_extraction_response(self, input_str: str) -> tuple[bool, str | None]:
        try:
            ExtractionResponse.model_validate_json(input_str)
            return True, None
        except json.JSONDecodeError as je:
            return False, f"Invalid JSON: {je}"
        except ValidationError as ve:
            first_error = ve.errors()[0]
            field = ".".join(str(x) for x in first_error["loc"])
            return False, f"{field}: {first_error["msg"]}"


    async def _prompts_for_first_phase(self) -> tuple[str, str]:
        _orig_list = await self.original_memory.get_messages(include_metadata=True)
        _tran_list = await self.translated_memory.get_messages(include_metadata=True)
        original_str = self.memory_parsing_to_string(_orig_list)
        translated_str = self.memory_parsing_to_string(_tran_list)


        user_prompt = RESOURCES.user_first_phase.render(
            translated_user_input=self.state.translated_user_input,
            original_conversation=original_str,
            translated_conversation=translated_str,
            documentation_context=RESOURCES.documentation_context
        )
        system_prompt = RESOURCES.system_first_phase
        print('=== STARTING PROMPTS ===')
        print(system_prompt, "\n")
        print(user_prompt, "\n")
        print("===ENDING PROMPTS===")
        return system_prompt, user_prompt


async def _prompts_for_first_phase_mock(
        original_memory: MessageStorage,
        translated_memory: MessageStorage,
):
    _orig_mock_list = await original_memory._get_user_memory()
    _orig_mock_list.messages.extend(fake_memory.taglish_original_history)
    orig_list = await original_memory.get_messages(include_metadata=True)
    original_str = IntentClassifier.memory_parsing_to_string(orig_list)

    _trans_mock_list = await translated_memory._get_user_memory()
    _trans_mock_list.messages.extend(fake_memory.taglish_translated_history)
    trans_list = await translated_memory.get_messages(include_metadata=True)
    translated_str = IntentClassifier.memory_parsing_to_string(trans_list)

    user_prompt = RESOURCES.user_first_phase.render(
        translated_user_input=fake_memory.taglish_user_input,
        original_conversation=original_str,
        translated_conversation=translated_str,
        documentation_context=RESOURCES.documentation_context
     )

    system_prompt = RESOURCES.system_first_phase
    print('=== STARTING PROMPTS ===')
    print( system_prompt, "\n")
    print( user_prompt, "\n")
    print("===ENDING PROMPTS===")
    return system_prompt, user_prompt


int_cla = IntentClassifier()
_inputs = {"user_id":"test","translated_user_input":fake_memory.taglish_user_input}
int_cla_kick = int_cla.kickoff_async(inputs=_inputs)
kick_resp = asyncio.run(int_cla_kick)
print(kick_resp)
