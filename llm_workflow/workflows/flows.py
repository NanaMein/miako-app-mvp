import json
import uuid
from typing import Any
from crewai.flow import Flow, start, listen, router, or_
from pydantic import BaseModel, ConfigDict
from llm_workflow.memory.short_term_memory.message_cache import MessageStorageV1
from llm_workflow.llm.groq_llm import GroqLLM
from fastapi import status, HTTPException
from llm_workflow.workflows.steps.language_step import LanguageFlow
from llm_workflow.workflows.steps.intent_step import IntentFlow


class EngineStates(BaseModel):
    input_message: str = ""
    input_user_id: str = ""
    model_config = ConfigDict(arbitrary_types_allowed=True)


class _AdaptiveChatbotEngine(Flow[EngineStates]):
    def __init__(self, **kwargs: Any):
        self._memory_storage: MessageStorageV1 | None = None
        super().__init__(**kwargs)
        self.chatbot = GroqLLM()



    @start()
    async def safety_content_moderator(self):
        pass #For development only, assume there is a content moderator here.


    @listen(safety_content_moderator)
    async def language_layer(self) -> dict[str, Any]:
        language_flow = LanguageFlow(
            user_id=self.state.input_user_id,
            original_message=self.state.input_message
        )
        return await language_flow.run()

    @listen(language_layer)
    async def intent_classifier(self, translation_response: dict[str, Any]) -> Exception | str:
        intent_flow = IntentFlow(
            user_id=self.state.input_user_id,
            input_data_obj=translation_response
        )
        return await intent_flow.run()

    @listen(intent_classifier)
    async def final_answer_test(self, data: Exception | str):
        if isinstance(data, Exception):
            return Exception(str(data))

        memory = await self.memory.get_messages(include_metadata=True)
        full_memory = json.dumps(memory)
        intents = data
        full_text = f"""===FULL CONVERSATION HISTORY===\n
        {full_memory}\n
        ===INTENTS===\n
        {intents}\n
        ===END===\n
        """
        return full_text

    @property
    def memory(self) -> MessageStorageV1:
        if self._memory_storage is None:
            self._memory_storage = MessageStorageV1(user_id=self.state.input_user_id)
        return self._memory_storage




class AdaptiveChatbot:
    def __init__(self, user_id: str | uuid.UUID | Any, input_message: str):
        self._engine: Flow[BaseModel] | None = None
        self._all_input_data = {
            "input_user_id": user_id,
            "input_message": input_message
        }


    @property
    def flow_engine(self) -> Flow[BaseModel]:
        if self._engine is None:
            self._engine = _AdaptiveChatbotEngine()
        return self._engine

    @property
    def _input_data(self):
        return self._all_input_data

    async def run(self) -> Any | str | None:
        try:
            response= await self.flow_engine.kickoff_async(inputs=self._input_data)
            if response is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Bad Request: {str(response)}")
            print("SSSSSSSSSSTART", response, "EEEEEEEEEEEEEEEND")
            return response
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

