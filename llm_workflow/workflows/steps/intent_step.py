from typing import Union, Any
from crewai.flow.flow import Flow, start, listen, and_, or_
from pydantic import BaseModel, ConfigDict
from llm_workflow.llm.groq_llm import GroqLLM, MODEL
from llm_workflow.prompts.prompt_library import IntentLibrary, DataExtractorLibrary
from llm_workflow.memory.short_term_memory.message_cache import MessageStorage
from groq.types.chat import ChatCompletionMessage
from jinja2 import Template
import asyncio


# SIMULATED CONVERSATION 1: TAGALOG/TAGLISH
# Context: User is skeptical about performance and cost.

taglish_conversation_history = [
    {
        "role": "user",
        "content": "Hi, tanong lang, mabilis ba talaga to? Kasi yung luma naming chatbot sobrang bagal.",
        "metadata": {"timestamp": "2023-10-27 10:00:00", "user_id": "pinoy_dev_01"}
    },
    {
        "role": "assistant",
        "content": "Opo! Miako handles thousands of users with sub-second latency kasi naka-Groq LPU siya. Hindi siya n8n na mabagal.",
        "metadata": {"timestamp": "2023-10-27 10:00:05", "model": "miako-v1"}
    },
    {
        "role": "user",
        "content": "Weh? Di nga? Pano pag sabay-sabay gumamit? Edi lag yan panigurado.",
        "metadata": {"timestamp": "2023-10-27 10:00:15", "user_id": "pinoy_dev_01"}
    },
    {
        "role": "assistant",
        "content": "Hindi po. Designed siya for high concurrency using Async FastAPI. May isolation din per user kaya walang lag.",
        "metadata": {"timestamp": "2023-10-27 10:00:20", "model": "miako-v1"}
    },
    {
        "role": "user",
        "content": "Ah ganun ba. Eh pano kung nagonline lahat, di ba mahal yun? Yung budget kasi namin limited lang.",
        "metadata": {"timestamp": "2023-10-27 10:00:35", "user_id": "pinoy_dev_01"}
    }
]

# FOR THE TRANSLATED MEMORY MOCK (What the LLM 'sees' as English)
taglish_translated_history = [
    {
        "role": "user",
        "content": "Hi, just asking, is this really fast? Because our old chatbot was very slow.",
        "metadata": {"timestamp": "2023-10-27 10:00:00"}
    },
    {
        "role": "assistant",
        "content": "Yes! Miako handles thousands of users with sub-second latency because it uses Groq LPU. It is not like n8n which is slow.",
        "metadata": {"timestamp": "2023-10-27 10:00:05"}
    },
    {
        "role": "user",
        "content": "Really? No way? What if everyone uses it at the same time? That would surely lag.",
        "metadata": {"timestamp": "2023-10-27 10:00:15"}
    },
    {
        "role": "assistant",
        "content": "No sir. It is designed for high concurrency using Async FastAPI. There is also isolation per user so there is no lag.",
        "metadata": {"timestamp": "2023-10-27 10:00:20"}
    },
    {
        "role": "user",
        "content": "Ah is that so. But what if everyone goes online, isn't that expensive? Our budget is limited.",
        "metadata": {"timestamp": "2023-10-27 10:00:35"}
    }
]
taglish_user_input = "Ah is that so. But what if everyone goes online, isn't that expensive? Our budget is limited."

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
            documentation_context="None"
        )
        system_prompt = RESOURCES.system_first_phase

        return system_prompt, user_prompt

    async def _prompts_for_first_phase_mock(self):
        _orig_mock_list = await self.original_memory._get_user_memory()
        _orig_mock_list.messages.extend(taglish_conversation_history)
        orig_list = await self.original_memory.get_messages(include_metadata=True)
        original_str = self.memory_parsing_to_string(orig_list)

        _trans_mock_list = await self.translated_memory._get_user_memory()
        _trans_mock_list.messages.extend(taglish_translated_history)
        trans_list = await self.translated_memory.get_messages(include_metadata=True)
        translated_str = self.memory_parsing_to_string(trans_list)

        user_prompt = RESOURCES.user_first_phase.render(
            translated_user_input=taglish_user_input,
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
