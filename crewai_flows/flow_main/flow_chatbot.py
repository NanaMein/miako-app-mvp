from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.cohere import CohereEmbedding
from collections import deque, defaultdict
from typing import Deque, Type, Optional, Tuple, TypeVar, Generic, Any
from llama_index.core.storage.chat_store.base_db import MessageStatus
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex, Document, StorageContext, SimpleDirectoryReader
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, AgentWorkflow
from llama_index.core.memory.memory import Memory
from llama_index.core.workflow import Context
from datetime import datetime, timezone, timedelta
from crewai.flow import Flow, start, listen, router, or_, and_, persist
from pydantic import BaseModel, Field, PrivateAttr
from dotenv import load_dotenv
from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import tool, BaseTool
from crewai.project import CrewBase, task, agent, crew
from cachetools import TTLCache, cached
from llama_index.core.schema import MetadataMode
from pymilvus import MilvusException, MilvusClient
import grpc
import os
import pytz
import time
from grpc.aio import AioRpcError

from crewai_flows.flow_main.query_chat_history import chat_conversation_history

load_dotenv()




class FlowMainStates(BaseModel):
    input_message: str = ""
    input_user_id: Optional[Any] = ""

class FlowMainWorkflow(Flow[FlowMainStates]):

    @start()
    def start_with_id(self):
        input_user_id = self.state.input_user_id
        input_string_id = str(input_user_id)
        return input_string_id


    @listen(start_with_id)
    def get_query_context(self, user_id):
        vector_context = chat_conversation_history(user_id=user_id, input_message=self.state.input_message)
        return vector_context
