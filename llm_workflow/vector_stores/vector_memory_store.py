import os
from llama_index.core import Document
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv
from llm_workflow.vector_stores.vector_connection import MilvusVectorStoreConnection
from llama_index.core.prompts import PromptTemplate


load_dotenv()


NODE_FORMAT_TEMPLATE="""
    <Node_start>\n
    <Node_text>{node_text}</Node_text>\n
    <Node_metadata>{node_metadata}</Node_metadata>\n
    <Node_score>{node_score}<Node_score>\n
    </Node_end>\n\n
"""
MESSAGE_FORMAT_TEMPLATE = """
    <conversation_turn>\n
        <user_turn>\n
            {{user_message}}\n
        </user_turn>\n
        <assistant_turn>\n
            {{assistant_message}}\n
        </assistant_turn>\n

    </conversation_turn>\n \n---/---/---\n"""

NODE_TEMPLATE = PromptTemplate(NODE_FORMAT_TEMPLATE)
MESSAGE_TEMPLATE = PromptTemplate(MESSAGE_FORMAT_TEMPLATE)



class ConversationMemoryStore:

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