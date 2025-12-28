import os
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.groq import Groq
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
from crewai_flows.flow_main.vector import MilvusVectorStoreClass


load_dotenv()

milvus = MilvusVectorStoreClass()

def embed_model_cohere(input_type: str = "") -> CohereEmbedding:

    if input_type == "doc":
        _input_type = "search_document"
    else:
        _input_type="search_query"

    embed_model = CohereEmbedding(
        model_name="embed-v4.0",
        api_key=os.getenv('CLIENT_COHERE_API_KEY'),
        input_type=_input_type
    )
    return embed_model

def llm_groq():
    return Groq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.getenv("CLIENT_GROQ_API_1"),
        temperature=0.5
    )

def get_query_engine(vector_store: MilvusVectorStore, embed_model: BaseEmbedding, llm: Groq) -> BaseQueryEngine:
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )
    query_engine = index.as_query_engine(
        vector_store_query_mode="hybrid", similarity_top_k=5, llm=llm
    )
    return query_engine

def chat_conversation_history(user_id: str, input_message: str) -> str:

    vector_store = milvus.get_vector_chat_history(user_id=user_id)

    embed_model = embed_model_cohere()

    llm = llm_groq()

    query_engine = get_query_engine(
        vector_store=vector_store,
        embed_model=embed_model,
        llm=llm
    )

    chat_result = query_engine.query(input_message)

    return chat_result.response

