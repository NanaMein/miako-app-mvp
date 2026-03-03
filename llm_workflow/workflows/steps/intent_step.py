import uuid
from typing import Union, Any, List
from crewai.flow.flow import Flow, start, listen, router, or_
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from llm_workflow.llm.groq_llm import GroqLLM, MODEL
from llm_workflow.prompts.prompt_library import IntentLibrary
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage, MessageStorageV1
from llm_workflow.memory.short_term_memory._fake_memory_testing import fake_memory
from groq.types.chat import ChatCompletionMessage
from jinja2 import Template
from dataclasses import dataclass
import asyncio
import json
import re





class Prompts:
    _intent_library = IntentLibrary()
    _data_extractor_base_template = _intent_library.get_prompt("user-prompt.data-extractor")
    _fact_validator_base_template = _intent_library.get_prompt("user-prompt.facts-validator")

    system_data_extractor = _intent_library.get_prompt("system-prompt.data-extractor")
    user_data_extractor_template = Template(_data_extractor_base_template, enable_async=True)

    system_fact_validator = _intent_library.get_prompt("system-prompt.facts-validator")
    user_fact_validator_template = Template(_fact_validator_base_template, enable_async=True)

    documentation_context = _intent_library.get_prompt("documentation-context")


PROMPTS_OLD = Prompts()

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


class _IntentClassifierOriginal(Flow[IntentState]):
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

            user_prompt = await PROMPTS_OLD.user_fact_validator_template.render_async(
                translated_user_input=self.state.translated_user_input,
                translated_conversation_history=translated_history,
                original_conversation_history=original_history,
                extracted_data_context=self.state.current_data_extraction
            )
            system_prompt = PROMPTS_OLD.system_fact_validator
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


        user_prompt = await PROMPTS_OLD.user_data_extractor_template.render_async(
            translated_user_input=self.state.translated_user_input,
            original_conversation=original_str,
            translated_conversation=translated_str,
            documentation_context=PROMPTS_OLD.documentation_context
        )
        system_prompt = PROMPTS_OLD.system_data_extractor
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
    original_str = _IntentClassifierOriginal.memory_parsing_to_string(orig_list)

    _trans_mock_list = await translated_memory._get_user_memory()
    _trans_mock_list.messages.extend(fake_memory.taglish_translated_history)
    trans_list = await translated_memory.get_messages(include_metadata=True)
    translated_str = _IntentClassifierOriginal.memory_parsing_to_string(trans_list)

    user_prompt = await PROMPTS_OLD.user_data_extractor_template.render_async(
        translated_user_input=fake_memory.taglish_user_input,
        original_conversation=original_str,
        translated_conversation=translated_str,
        documentation_context=PROMPTS_OLD.documentation_context
    )

    system_prompt = PROMPTS_OLD.system_data_extractor
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


class IntentFlowTemporary:
    def __init__(
        self,
        user_id: Union[str, Any],
        translated_user_input: str = "",
        original_user_input: str = ""
    ):
        self.translated_user_input = translated_user_input
        self.original_user_input = original_user_input
        self.user_id = user_id
        self.flow = _IntentClassifierOriginal()

    async def run(self):
        flow = await self.flow.kickoff_async(inputs={
            "user_id": self.user_id,
            "translated_user_input": self.translated_user_input,
            "original_user_input": self.original_user_input
        })
        if isinstance(flow, Exception):
            raise flow

        return flow

@dataclass(slots=True)
class LanguageObject:
    translated_text: str
    original_text: str
    created_at: str
    source_language: str



class PromptsV1:
    def __init__(self):
        self._doc_context = None
        self._user_data_extractor_template = None
        self._intent_library = None
        self._system_data_extractor_prompt = None
        self._user_facts_validator_template = None
        self._system_facts_validator_prompt = None

    @property
    def intent(self):
        if self._intent_library is None:
            self._intent_library = IntentLibrary()
        return self._intent_library

    @property
    def documentation_context(self):
        if self._doc_context is None:
            self._doc_context  = self.intent.get_prompt("documentation-context")
        return self._doc_context

    @property
    def _get_user_data_extractor_template(self) -> Template:
        if self._user_data_extractor_template is None:
            _prompt = self.intent.get_prompt("v1.data-extractor.user-prompt")
            _template = Template(_prompt, enable_async=True)
            self._user_data_extractor_template = _template
        return self._user_data_extractor_template

    @property
    def _get_user_facts_validator_template(self) -> Template:
        if self._user_facts_validator_template is None:
            _prompt = self.intent.get_prompt("v1.facts-validator.user-prompt")
            _template = Template(_prompt, enable_async=True)
            self._user_facts_validator_template = _template
        return self._user_facts_validator_template

    @property
    def system_data_extractor(self) -> str:
        if self._system_data_extractor_prompt is None:
            self._system_data_extractor_prompt = self.intent.get_prompt("v1.data-extractor.system-prompt")
        return self._system_data_extractor_prompt

    async def user_data_extractor(self, input_obj_data: dict[str, Any], history: MessageStorageV1) -> str:
        conversation_history = await history.get_messages(include_metadata=True)
        lang_obj = LanguageObject(**input_obj_data)
        template = self._get_user_data_extractor_template
        user_prompt = await template.render_async(
            translated_text=lang_obj.translated_text,
            original_text=lang_obj.original_text,
            created_at=lang_obj.created_at,
            source_language=lang_obj.source_language,
            documentation_context=self.documentation_context,
            conversation_history=conversation_history
        )
        return user_prompt

    @property
    def system_facts_validator(self):
        if self._system_facts_validator_prompt is None:
            self._system_facts_validator_prompt = self.intent.get_prompt("v1.facts-validator.system-prompt")
        return self._system_facts_validator_prompt


    async def user_facts_validator(self, extracted_data:str, history: MessageStorageV1, input_obj_data: dict[str, Any]) -> str:
        lang_obj = LanguageObject(**input_obj_data)
        _template = self._get_user_facts_validator_template
        conversation_history = await history.get_messages(include_metadata=True)
        user_prompt = await _template.render_async(
            translated_user_input=lang_obj.translated_text,
            conversation_history=conversation_history,
            extracted_data_context=extracted_data
        )
        return user_prompt

PROMPTS = PromptsV1()



class IntentFlowStates(BaseModel):
    user_id: str | uuid.UUID | Any = ""
    data_input: dict[str, Any] = Field(default_factory=dict)
    data_extraction_handler: str = ""
    error_exception: Exception | str | None = None
    facts_validation_handler: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)



class _IntentClassifier(Flow[IntentFlowStates]):
    def __init__(self, **kwargs):
        self._message_storage_v1: MessageStorageV1 | None = None
        super().__init__(**kwargs)

    @property
    def memory(self):
        if self._message_storage_v1 is None:
            self._message_storage_v1 = MessageStorageV1(self.state.user_id)
        return self._message_storage_v1

    @property
    def groq_llm(self):
        return GroqLLM()

    @start()
    async def start_or_testing_phase(self):
        pass

    @listen(start_or_testing_phase)
    async def prep_prompts(self) -> tuple[str, str]:
        system_prompt = PROMPTS.system_data_extractor
        user_prompt = await PROMPTS.user_data_extractor(
            input_obj_data=self.state.data_input,
            history=self.memory
        )
        return system_prompt, user_prompt

    @listen(prep_prompts)
    async def start_with_data_extraction(self, data_prompts: tuple[str, str]) -> str | Exception:
        try:
            system_prompt, user_prompt = data_prompts

            data_extraction_llm = self.groq_llm
            data_extraction_llm.add_system(system_prompt)
            data_extraction_llm.add_user(user_prompt)

            return await data_extraction_llm.groq_chat(
                model=MODEL.scout, temperature=.1,
                max_completion_tokens=8000
            )

        except Exception as ex:
            return ex

    @listen(start_with_data_extraction)
    def validating_extracted_data(self, input_data: str | Exception) -> str | None:
        if isinstance(input_data, Exception):
            self.state.error_exception = input_data
            return None

        if isinstance(input_data, str):
            is_valid, parsed_data, error = self._validating_data_extraction_response(input_data)

            if is_valid:
                return json.dumps(parsed_data)

            else:
                self.state.error_exception = error
                return None

        self.state.error_exception = "Unexpected error in validation"
        return None

    @listen(validating_extracted_data)
    def updating_states(self, data: str | None):
        if data is not None:
            self.state.data_extraction_handler = data
            return True
        else:
            return False


    @router(updating_states)
    def data_extraction_router(self, data: bool):

        if data:
            return "DATA_EXTRACTION_PASSED"
        else:
            return "DATA_EXTRACTION_ERROR"

    @listen("DATA_EXTRACTION_PASSED")
    async def generating_prompts_for_validator(self):
        try:
            extracted_data = self.state.data_extraction_handler
            user_prompt = await PROMPTS.user_facts_validator(
                history=self.memory,
                input_obj_data=self.state.data_input,
                extracted_data=extracted_data
            )
            system_prompt = PROMPTS.system_facts_validator
            return system_prompt, user_prompt
        except Exception as ex:
            return ex

    @listen("DATA_EXTRACTION_ERROR")
    def data_extraction_failure(self):
        return False


    @listen(generating_prompts_for_validator)
    async def facts_validator(self, _prompts: tuple[str, str] | Exception):
        if isinstance(_prompts, Exception):
            self.state.error_exception = _prompts
            return False
        try:
            system_prompt, user_prompt = _prompts
            llm = self.groq_llm
            llm.add_system(system_prompt)
            llm.add_user(user_prompt)
            facts_response = await llm.groq_chat(
                model=MODEL.gpt_oss_120,
                reasoning_effort="low",
                temperature=.1,
                max_completion_tokens=20_000
            )

            self.state.facts_validation_handler = facts_response
            return True

        except Exception as ex:
            self.state.error_exception = ex
            return False

    @router(facts_validator)
    def facts_validation_router(self, data):
        if data:
            return "FACTS_VALIDATED"
        else:
            return "FACTS_VALIDATION_ERROR"

    @listen("FACTS_VALIDATED")
    def facts_validation_passed(self):
        return True

    @listen("FACTS_VALIDATION_ERROR")
    def facts_validation_failure(self):
        return False


    @listen(or_(facts_validation_passed, facts_validation_failure, data_extraction_failure))
    def finalizing_output(self, is_passed: bool) -> Exception | str:
        if is_passed:
            return self.state.facts_validation_handler
        else:
            ex = Exception(f"Flow error output: {str(self.state.error_exception)}")
            return ex



    def _validating_data_extraction_response(self, input_str: str) -> tuple[bool, dict | None, str | None]:
        if not input_str:
            return False, None, "Empty LLM Response"

        try:
            data = json.loads(input_str)

        except json.JSONDecodeError:
            extracted = self.extract_json(input_str)

            if not extracted:
                return False, None, "No Json found in response"

            try:
                data = json.loads(extracted)
            except Exception as ex:
                return False, None, f"Json decode failed after extraction: {ex}"

        except Exception as _ex:
            return False, None, f"Unexpected error: {str(_ex)}"



        if data.get("message") in ['FORMAT_ERROR', 'NO_RELEVANT_CONTEXT']:
            return True, data, None


        try:
            ExtractionResponse.model_validate(data)
            return True, data, None

        except json.JSONDecodeError as je:
            return False, None, f"Invalid JSON: {je}"

        except ValidationError as ve:
            first_error = ve.errors()[0]
            field = ".".join(str(x) for x in first_error["loc"])
            return False, None, f"{field}: {first_error["msg"]}"



    @staticmethod
    def extract_json(text: str) -> str | None:
        if not text:
            return None
        text = text.strip()

        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```", "", text)

        stack = []
        _start = None

        for i, char in enumerate(text):
            if char == "{":
                if _start is None:
                    _start = i
                stack.append("{")

            elif char == "}":
                if stack:
                    stack.pop()
                    if not stack and _start is not None:
                        return text[_start:i + 1]

        return None

class IntentFlow:
    def __init__(
            self,
            user_id: str | uuid.UUID | Any,
            input_data_obj: dict[str, Any],
    ):
        self._flow: Flow[BaseModel] | None = None
        self._input = {
            "user_id":user_id,
            "data_input":input_data_obj,
        }

    @property
    def flow(self):
        if self._flow is None:
            self._flow = _IntentClassifier()
        return self._flow

    async def run(self):
        return await self.flow.kickoff_async(inputs=self._input)

#
# _metadata = {
#     "original_text": "original text test",
#     "created_at": "created at test",
#     "source_language": "source language test",
#     "translated_text": "translation text test"
# }
# # test_func = _PROMPTS.user_data_extractor(_metadata)
# # test_user_prompt = asyncio.run(test_func)
# # print(test_user_prompt)
#
#
# async def intent_flow():
#     _inputs = {
#         "user_id": str(uuid.uuid4()),
#         "data_input": _metadata
#     }
#     _intent = _InternalIntentClassifier()
#     _intent.plot()
#     return await _intent.kickoff_async(inputs=_inputs)
#
#
# test_flow_intent = asyncio.run(intent_flow())
# print(test_flow_intent)