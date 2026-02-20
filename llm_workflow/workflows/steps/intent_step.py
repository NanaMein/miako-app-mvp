from typing import Union, Any
from crewai.flow.flow import Flow, start, listen, and_, or_
from pydantic import BaseModel, ConfigDict
from llm_workflow.llm.groq_llm import GroqLLM, MODEL
from llm_workflow.prompts.prompt_library import DataExtractorLibrary
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage
from llm_workflow.memory.short_term_memory._fake_memory_testing import fake_memory
from groq.types.chat import ChatCompletionMessage
from jinja2 import Template
import asyncio


class AppResources:
    _data_extractor_prompts = DataExtractorLibrary()
    _user_first_phase_template = _data_extractor_prompts.get_prompt("user-first-phase")

    system_first_phase = _data_extractor_prompts.get_prompt("system-first-phase")
    user_first_phase = Template(_user_first_phase_template)
    documentation_context = _data_extractor_prompts.get_prompt("documentation-context")

RESOURCES = AppResources()



class IntentState(BaseModel):
    user_id: Union[str, Any] = ""
    translated_user_input: str = ""
    original_user_input: str = ""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class IntentClassifier(Flow[IntentState]):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.llm = GroqLLM()


    @start()
    async def start_with_data_extraction(self):
        system_prompt, user_prompt = await self._prompts_for_first_phase_mock()
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
        return response


    @property
    def original_memory(self):
        _user_id = f"original_x_{self.state.user_id}"
        return MessageStorage(user_id=_user_id)

    @property
    def translated_memory(self):
        _user_id = f"translated_x_{self.state.user_id}"
        return MessageStorage(user_id=_user_id)



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

        return system_prompt, user_prompt

    async def _prompts_for_first_phase_mock(self):
        _orig_mock_list = await self.original_memory._get_user_memory()
        _orig_mock_list.messages.extend(fake_memory.taglish_original_history)
        orig_list = await self.original_memory.get_messages(include_metadata=True)
        original_str = self.memory_parsing_to_string(orig_list)

        _trans_mock_list = await self.translated_memory._get_user_memory()
        _trans_mock_list.messages.extend(fake_memory.taglish_translated_history)
        trans_list = await self.translated_memory.get_messages(include_metadata=True)
        translated_str = self.memory_parsing_to_string(trans_list)

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

x = IntentClassifier()
_xx = x.kickoff_async()
xxx = asyncio.run(_xx)
print(xxx)
