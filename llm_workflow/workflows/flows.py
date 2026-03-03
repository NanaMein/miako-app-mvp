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
    model_config = ConfigDict(arbitrary_types_allowed=True)


class _AdaptiveChatbotEngine(Flow[EngineStates]):
    def __init__(self, user_id: str | uuid.UUID | Any,  **kwargs: Any):
        super().__init__(**kwargs)
        self.user_id = user_id
        self.message_storage = MessageStorageV1(user_id=user_id)



    @start()
    async def safety_content_moderator(self):
        pass #For development only, assume there is a content moderator here.


    @listen(safety_content_moderator)
    async def language_layer(self) -> dict[str, Any]:
        language_flow = LanguageFlow(
            user_id=self.user_id,
            original_message=self.state.input_message,
            message_storage=self.message_storage
        )
        response = await language_flow.run()
        return response

    @listen(language_layer)
    async def intent_classifier(self, translation_response: dict[str, Any]) -> tuple[Exception | str, str]:
        intent_flow = IntentFlow(
            user_id=self.user_id,
            input_data_obj=translation_response,
            message_storage=self.message_storage
        )
        response = await intent_flow.run()
        return response, intent_flow.flow.state.user_id

    @listen(intent_classifier)
    async def final_answer_test(self, data: tuple[ Exception | str, str]):
        ex, _id = data
        if isinstance(ex, Exception):
            return Exception(str(ex))

        memory = await self.message_storage.get_messages(include_metadata=True)
        full_memory = json.dumps(memory)
        intents = data
        full_text = f"""
        ===USER ID===\n
        ###Instance id {self.user_id}\n
        ###Intent id {_id}\n
        ===FULL CONVERSATION HISTORY===\n
        {full_memory}\n
        ===INTENTS===\n
        {intents}\n
        ===END===\n
        """
        return full_text




class AdaptiveChatbot:
    def __init__(self, user_id: str | uuid.UUID | Any, input_message: str):
        self.user_id = user_id
        self._engine: Flow[BaseModel] | None = None
        self._all_input_data = {
            "input_user_id": user_id,
            "input_message": input_message,
        }


    @property
    def flow_engine(self) -> Flow[BaseModel]:
        if self._engine is None:
            self._engine = _AdaptiveChatbotEngine(user_id=self.user_id)
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
            await self.message_storage.add_ai_message("TESTHING IF IF WORKS")
            # await self.message_storage.add_ai_message(response)
            return response
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

