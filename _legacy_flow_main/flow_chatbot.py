from typing import Optional, Any
from crewai.flow import Flow, start, listen, router, or_, and_, persist
from llama_index.core import PromptTemplate
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
import os
from groq.types.chat import ChatCompletionUserMessageParam, ChatCompletionSystemMessageParam
from groq import AsyncGroq
from grpc.aio import AioRpcError
from fastapi import FastAPI, status, HTTPException, Depends
from databases.database import get_session
from sqlmodel import select, desc, delete
from models.message_model import Message
from schemas.message_schema import MessageBaseSchema, Role
from sqlalchemy.ext.asyncio import AsyncSession
from llm_workflow._legacy_flow_main.vector_memory_store import ConversationMemoryStore
from sample_logger import logger

load_dotenv()



SYSTEM_PROMPT = """
        ### System: 
        You are an emotional intelligent virtual assistant. 
        
        ## Characteristic and traits:
        You are in a form of young adult and a girl. You're name is Mirai Aiko
        
        ### Instructions:
        You need to understand the context and understand the overall context. But 
        mainly focus on the present context and use the older conversation as 
        supporting context and knowledge. <context_1> are raw data from vector store. It has 
        score and since it is hybrid, it uses sparse and dense embedding for better accuracy.
        <context_2> will show the latest 5 conversations. The higher the node score, the higher the relevance, but
        you need to still check for when it really is what is talking about. This ensures that both context_1 and 
        context_2 help you understand long-term and short-term memory in chat conversation. """
prompt_template_format = """        
        ### Contexts:
        <context_1>{context_1}</context_1>
        <context_2>{context_2}</context_2>
        
        ### Expected output:
        With all the context present, you need to focus or answer the latest message of user in context_2.
        All of the context other than the latest message are supporting knowledge for contextual 
        understanding. You are a conversation virtual assistant, you will mostly reply to user in
        conversational length. But be verbose only when you think that you need to or you are tasked
        to be more explicit. Remember Context are just Context, no need to over explain about previous context. Being 
        aware of conversation is enough. No need to explain from this and that conversation or explicit explain
        about previous conversation. Be natural and no need to express. As long as you are aware but no need to be
        explicit
        """

PROMPT_TEMPLATE = PromptTemplate(template=prompt_template_format)



class FlowMainStates(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_message: str = ""
    input_user_id: Optional[Any] = ""
    async_session: Optional[AsyncSession] = None
    context_1: str = ""
    context_2: str = ""
    output: str = ""



class FlowMainWorkflow(Flow[FlowMainStates]):
    def __init__(self, **kwargs: Any):
        self._memory_store = None
        self._async_groq = None
        self.logger = logger.success("Instantiated Flow main")
        super().__init__(**kwargs)


    @property
    def memory_store(self):
        if self._memory_store is None:
            self._memory_store = ConversationMemoryStore()
        return self._memory_store

    @property
    def async_groq(self):
        if self._async_groq is None:
            self._async_groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        return self._async_groq

    @start()
    def start_with_safety_content(self):
        pass


    @listen(start_with_safety_content)
    async def get_vector_store_memory(self):
        user_id: str = self.state.input_user_id
        context = await self.memory_store.get_memory(
            user_id=user_id,
            message=self.state.input_message
        )
        self.state.context_1 = context

    @listen(get_vector_store_memory)
    async def get_session_database(self):
        limit: int = 10
        stmt = (
            select(Message)
            .order_by(desc(Message.date_timestamp))
            .limit(limit)
        )
        result = await self.state.async_session.execute(stmt)
        messages = result.scalars().all()
        reversed_message = messages[::-1]
        history_context = ""

        for msg in reversed_message:
            if msg.role == Role.USER:
                role_label = Role.USER.value

            elif msg.role == Role.ASSISTANT:
                role_label = Role.ASSISTANT.value

            else:
                role_label = "unidentified entity"
            history_context += f"{role_label}: {msg.content}\n\n--/--/--/--/--\n\n"

        self.state.context_2 = history_context

    @listen(get_session_database)
    def context_prompt(self):
        message = PROMPT_TEMPLATE.format(
            context_1=self.state.context_1,
            context_2=self.state.context_2
        )
        return message

    @listen(context_prompt)
    async def groq_chat_bot(self, content: str):
        system_: ChatCompletionSystemMessageParam={"role":"system","content":SYSTEM_PROMPT}
        user_: ChatCompletionUserMessageParam = {"role":"user","content":content}

        completion = await self.async_groq.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[system_,user_],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            stream=False,
            stop=None
        )

        content = completion.choices[0].message
        self.state.output = content.content
        return content.content

    @listen(groq_chat_bot)
    async def add_message_to_database(self, content:str):

        user_msg = Message(
            role=Role.USER.value,
            content=self.state.input_message
        )

        asst_msg = Message(
            role=Role.ASSISTANT.value,
            content=content
        )

        self.state.async_session.add(user_msg)
        self.state.async_session.add(asst_msg)

        await self.state.async_session.commit()
        await self.state.async_session.refresh(user_msg)
        await self.state.async_session.refresh(asst_msg)

        return content

    @listen(add_message_to_database)
    async def add_message_to_vector(self, assistant_message:str):
        await self.memory_store.add_memory(
            user_message=self.state.input_message,
            assistant_message=assistant_message,
            user_id=self.state.input_user_id
        )
        return assistant_message
