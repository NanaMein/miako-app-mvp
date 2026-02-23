from typing import Union, Any, List
from crewai.flow.flow import Flow, start, listen, router
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from llm_workflow.llm.groq_llm import GroqLLM, MODEL
from llm_workflow.prompts.prompt_library import IntentLibrary
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage
from llm_workflow.memory.short_term_memory._fake_memory_testing import fake_memory
from groq.types.chat import ChatCompletionMessage
from jinja2 import Template
import asyncio
import json





class Prompts:
    _intent_library = IntentLibrary()
    _data_extractor_base_template = _intent_library.get_prompt("user-prompt.data-extractor")
    _fact_validator_base_template = _intent_library.get_prompt("user-prompt.facts-validator")

    system_data_extractor = _intent_library.get_prompt("system-prompt.data-extractor")
    user_data_extractor_template = Template(_data_extractor_base_template, enable_async=True)

    system_fact_validator = _intent_library.get_prompt("system-prompt.facts-validator")
    user_fact_validator_template = Template(_fact_validator_base_template, enable_async=True)

    documentation_context = _intent_library.get_prompt("documentation-context")


PROMPTS = Prompts()

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
    current_data_extraction: str = ""
    current_fact_validation: str = ""
    error_exception: Exception | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class _IntentClassifier(Flow[IntentState]):
    def __init__(self, **kwargs: Any):
        self._translated_memory: MessageStorage | None = None
        self._original_memory: MessageStorage | None = None
        self.in_development_phase: bool = False
        super().__init__(**kwargs)
        self.extraction_worker = GroqLLM()
        self.validator_worker = GroqLLM()



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
        try:
            system_prompt, user_prompt = prompts
            self.extraction_worker.add_system(system_prompt)
            self.extraction_worker.add_user(user_prompt)
            response = await self.extraction_worker.groq_message_object(model=MODEL.scout, return_as_object=True, temperature=.1)
            return response
        except Exception as ex:
            return ex

    @listen(start_with_data_extraction)
    def validating_extracted_data(self, _resp):
        if isinstance(_resp, Exception):
            self.state.error_exception = _resp
            return "error_data_extraction"

        if isinstance(_resp, ChatCompletionMessage):
            response = _resp.content
        else:
            response = str(_resp)

        is_valid, error = self._validating_data_extraction_response(response)
        if is_valid:

            self.state.current_data_extraction = response
            return response

        else:
            self.state.latest_error_catch = error
            return "error_data_extraction"

    @router(validating_extracted_data)
    def data_extraction_router(self, data: str):
        if data == "error_data_extraction":
            return "ERROR"

        else:
            return "DATA_EXTRACTION_PASSED"

    @listen("DATA_EXTRACTION_PASSED")
    async def generating_prompts_for_validator(self):
        try:
            translated_history = await self.translated_memory.get_messages(include_metadata=True)
            original_history = await self.original_memory.get_messages(include_metadata=True)

            user_prompt = await PROMPTS.user_fact_validator_template.render_async(
                translated_user_input=self.state.translated_user_input,
                translated_conversation_history=translated_history,
                original_conversation_history=original_history,
                extracted_data_context=self.state.current_data_extraction
            )
            system_prompt = PROMPTS.system_fact_validator
            return system_prompt, user_prompt
        except Exception as ex:
            return ex

    @listen(generating_prompts_for_validator)
    async def facts_validator(self, _prompts: tuple[str, str] | Exception):
        if isinstance(_prompts, Exception):
            self.state.error_exception = _prompts
            return "error_fact_validation"
        try:
            system_prompt, user_prompt = _prompts
            self.validator_worker.add_system(system_prompt)
            self.validator_worker.add_user(user_prompt)
            facts_response = await self.validator_worker.groq_chat(
                model=MODEL.maverick, temperature=.1
            )
            self.state.current_fact_validation = facts_response
            return facts_response

        except Exception as ex:
            self.state.error_exception = ex
            return "error_fact_validation"

    @router(facts_validator)
    def fact_validator_router(self, facts_response: str):
        if facts_response == "error_fact_validation":
            return "ERROR"
        else:
            return "FACT_VALIDATION_PASSED"

    @listen("FACT_VALIDATION_PASSED")
    def validation_passed(self):
        return self.state.current_fact_validation

    @listen("ERROR")
    def error_exception_catcher(self):
        return self.state.error_exception


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
            metadata = msg.get("metadata", "No Metadata Available")

            if metadata and metadata != "No Metadata Available":
                msg_str = f"{role}:\n{content}\nMetadata:\n{metadata}\n===\n"
            else:
                msg_str = f"{role}:\n{content}\n===\n"

            _list.append(msg_str)
        full_str = "".join(_list)
        return full_str

    @staticmethod
    def _validating_data_extraction_response(input_str: str) -> tuple[bool, str | None]:
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


        user_prompt = await PROMPTS.user_data_extractor_template.render_async(
            translated_user_input=self.state.translated_user_input,
            original_conversation=original_str,
            translated_conversation=translated_str,
            documentation_context=PROMPTS.documentation_context
        )
        system_prompt = PROMPTS.system_data_extractor
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
    original_str = _IntentClassifier.memory_parsing_to_string(orig_list)

    _trans_mock_list = await translated_memory._get_user_memory()
    _trans_mock_list.messages.extend(fake_memory.taglish_translated_history)
    trans_list = await translated_memory.get_messages(include_metadata=True)
    translated_str = _IntentClassifier.memory_parsing_to_string(trans_list)

    user_prompt = await PROMPTS.user_data_extractor_template.render_async(
        translated_user_input=fake_memory.taglish_user_input,
        original_conversation=original_str,
        translated_conversation=translated_str,
        documentation_context=PROMPTS.documentation_context
    )

    system_prompt = PROMPTS.system_data_extractor
    print('=== STARTING PROMPTS ===')
    print( system_prompt, "\n")
    print( user_prompt, "\n")
    print("===ENDING PROMPTS===")
    return system_prompt, user_prompt

# int_cla = IntentClassifier()
# _inputs = {"user_id":"test","translated_user_input":fake_memory.taglish_user_input}
# int_cla_kick = int_cla.kickoff_async(inputs=_inputs)
# kick_resp = asyncio.run(int_cla_kick)
# print(kick_resp)


class IntentFlow:
    def __init__(
        self,
        user_id: Union[str, Any],
        translated_user_input: str = "",
        original_user_input: str = ""
    ):
        self.translated_user_input = translated_user_input
        self.original_user_input = original_user_input
        self.user_id = user_id
        self.flow = _IntentClassifier()

    async def run(self):
        flow = await self.flow.kickoff_async(inputs={
            "user_id": self.user_id,
            "translated_user_input": self.translated_user_input,
            "original_user_input": self.original_user_input
        })
        if isinstance(flow, Exception):
            raise flow

        return flow