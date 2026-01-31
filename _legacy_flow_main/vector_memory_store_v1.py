from typing import Literal
from fastapi import HTTPException, status
from llama_index.core import Document
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex
from llm_workflow.vector_stores.vector_connection import MilvusVectorStoreConnection
from llm_workflow.config_files.config import workflow_settings
from llama_index.core.prompts import PromptTemplate
from datetime import datetime, timezone



PRESENTATION_NODE_TEMPLATE = """
<Node>
  <text>{text}</text>
  <source>{source}</source>
  <turn_index>{turn_index}</turn_index>
  <score>{score}</score>
</Node>
"""
PRESENTATION_MESSAGE_TEMPLATE = """
<conversation_turn>
  <user_turn>{user_message}</user_turn>
  <assistant_turn>{assistant_message}</assistant_turn>
</conversation_turn>
---/---/---
"""

PRESENTATION_NODE = PromptTemplate(PRESENTATION_NODE_TEMPLATE)
PRESENTATION_MESSAGE = PromptTemplate(PRESENTATION_MESSAGE_TEMPLATE)

EMBED_TYPE = Literal["document", "query"]
SPLITTER = SentenceSplitter(chunk_size=360, chunk_overlap=60)


class ConversationMemoryStore:

    def __init__(self, user_id: str):
        self._user_id = user_id


    @property
    def milvus_store(self) -> MilvusVectorStoreConnection:
        return MilvusVectorStoreConnection(user_id=self._user_id)

    @staticmethod
    def embed_model_document():
        _embed_model_document = CohereEmbedding(
            model_name="embed-v4.0",
            api_key=workflow_settings.COHERE_API_KEY.get_secret_value(),
            input_type="search_document"
        )
        return _embed_model_document

    @staticmethod
    def embed_model_query():
        _embed_model_query = CohereEmbedding(
            model_name="embed-v4.0",
            api_key=workflow_settings.COHERE_API_KEY.get_secret_value(),
            input_type="search_query"
        )
        return _embed_model_query

    async def _get_index(self, embed_type: EMBED_TYPE):
        if embed_type not in ("document", "query"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="embed_type must be either 'document' or 'query'"
            )

        vector_store = await self.milvus_store.get_vector_store()
        if embed_type == "document":
            embed_model = ConversationMemoryStore.embed_model_document()
        else:
            embed_model = ConversationMemoryStore.embed_model_query()

        _index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=embed_model, use_async=True
        )
        return _index

    async def _get_retriever(self) -> BaseRetriever:
        index = await self._get_index(embed_type="query")

        retriever = index.as_retriever(
            vector_store_query_mode="hybrid", similarity_top_k=5
        )
        return retriever

    async def show_(self, query: str) -> str:

        retriever = await self._get_retriever()
        node_with_score = await retriever.aretrieve(query)

        output = ""

        for node in node_with_score:
            text = node.text
            md = getattr(node, "metadata", {}) or {}
            presentation = PRESENTATION_NODE.format(
                text=text,
                source=md.get("source","unknown"),
                turn_index=md.get("turn_index",-1),
                score=getattr(node, "score", 0),
            )
            output += presentation

        return output

    async def add_(
            self,
            user_message: str = "",
            assistant_message: str = "",
    ) -> bool:

        presentation_message = PRESENTATION_MESSAGE.format(
            user_message=user_message,
            assistant_message=assistant_message
        )

        plain_text_for_embed = f"User: {user_message}\nAssistant: {assistant_message}"

        _docs = [
            Document(
                text=plain_text_for_embed,
                metadata={
                    "presentation":presentation_message,
                    "type": "conversation_turn",
                    "user_message": user_message,
                    "assistant_message": assistant_message,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        ]

        index = await self._get_index(embed_type="document")

        try:
            nodes = await SPLITTER.aget_nodes_from_documents(_docs)

            await index.ainsert_nodes(nodes=nodes)
            return True
        except Exception as ex:
            print(f"error: {ex}")
            return False



class ConversationMemoryStoreOriginal:

    def __init__(self,node_template=NODE_TEMPLATE, message_template=MESSAGE_TEMPLATE):
        self._milvus = None
        self._embed_model_document = None
        self._embed_model_query = None
        self.node_template = node_template
        self.message_template = message_template
        self.sentence_splitter = SentenceSplitter(chunk_size=360, chunk_overlap=60)



    @property
    def milvus(self) -> MilvusVectorStoreConnection:
        if self._milvus is None:
            self._milvus = MilvusVectorStoreConnection() #NEEDS USER ID, MAKE NEW ISSUE AND PR HERE
        return self._milvus

    @property
    def embed_model_document(self) -> CohereEmbedding:
        if self._embed_model_document is None:

            self._embed_model_document = CohereEmbedding(
                model_name="embed-v4.0",
                api_key=os.getenv('CLIENT_COHERE_API_KEY'),
                input_type="search_document"
            )
        return self._embed_model_document

    @property
    def embed_model_query(self) -> CohereEmbedding:
        if self._embed_model_query is None:
            self._embed_model_query = CohereEmbedding(
                model_name="embed-v4.0",
                api_key=os.getenv('CLIENT_COHERE_API_KEY'),
                input_type="search_query"
            )
        return self._embed_model_query


    async def _get_vector(self, user_id:str) -> MilvusVectorStore:
        return await self.milvus.get_vector_store() #DOESNT NEED USER ID HERE, NEED ISSUE AND PR


    def _get_retriever(self, vector_store: MilvusVectorStore) -> BaseRetriever:

        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=self.embed_model_query, use_async=True
        )
        retriever = index.as_retriever(
            vector_store_query_mode="hybrid", similarity_top_k=5
        )
        return retriever

    async def get_memory(self, user_id: str, message: str) -> str:
        vector_store = await self._get_vector(user_id)
        retriever = self._get_retriever(vector_store)
        node_with_score = await retriever.aretrieve(message)

        retrieved_context = ""

        for node in node_with_score:
            template_with_node = self.node_template.format(
                node_text=node.text,
                node_metadata=node.metadata,
                node_score=node.score
            )
            retrieved_context += template_with_node

        return retrieved_context

    async def add_memory(
            self, user_id:str,
            user_message: str = "",
            assistant_message: str = "",
    ) -> bool:

        rendered_template = self.message_template.format(
            user_message=user_message,
            assistant_message=assistant_message
        )
        docs = [Document(text=rendered_template)]


        vector_store = await self._get_vector(user_id=user_id)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=self.embed_model_document, use_async=True
        )

        try:
            nodes = await self.sentence_splitter.aget_nodes_from_documents(docs)

            await index.ainsert_nodes(nodes=nodes)
            return True
        except Exception as ex:
            print(f"error: {ex}")
            return False