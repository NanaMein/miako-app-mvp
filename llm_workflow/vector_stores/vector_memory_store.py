from typing import Literal
from fastapi import HTTPException, status
from llama_index.core import Document
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.core import VectorStoreIndex
from llm_workflow.vector_stores.vector_connection import MilvusVectorStoreConnection
from llm_workflow.config_files.config import workflow_settings
from llama_index.core.prompts import PromptTemplate
from datetime import datetime, timezone
from llm_workflow.prompts.prompt_library import PromptLibrary
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PATH_DIR=str(BASE_DIR / "prompts" / "prompts.yaml")


LIB = PromptLibrary(file_path=PATH_DIR)

PRESENTATION_NODE = PromptTemplate(LIB.get_prompt("template.for_node"))
PRESENTATION_MESSAGE = PromptTemplate(LIB.get_prompt("template.for_message"))

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


class MemoryStoreException(HTTPException):
    def __init__(self, detail: str = "Error occurred during conversation memory."):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)