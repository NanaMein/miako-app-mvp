from typing import Optional
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import  BM25BuiltInFunction#,BGEM3SparseEmbeddingFunction
from dotenv import load_dotenv
from cachetools import TTLCache, LRUCache
from pymilvus import AsyncMilvusClient
from fastapi import HTTPException, status
import os
import asyncio


load_dotenv()

class MilvusVectorStoreClassAsync:

    def __init__(self):
        self.cache = LRUCache(maxsize=100)
        self.master_lock = asyncio.Lock()
        self.user_locks = {}
        # self._bgem3function = None

    # @property
    # def bgem3function(self) -> BGEM3SparseEmbeddingFunction:
    #     if self._bgem3function is None:
    #         self._bgem3function = BGEM3SparseEmbeddingFunction()
    #     return self._bgem3function

    @property
    def bm25function(self) -> BM25BuiltInFunction:
        return BM25BuiltInFunction()

    def user_id_to_collection_name(self, user_id: str) -> str:
        len_of_16_str = user_id.strip()[:16]
        return f"Collection_Of_{len_of_16_str}_2025_2026"

    async def _milvus_client(self) -> AsyncMilvusClient:
        client = AsyncMilvusClient(
            uri=os.getenv('CLIENT_URI'),
            token=os.getenv('CLIENT_TOKEN')
        )
        return client

    def _vector_store_with_bgem3(self, collection_name: str) -> MilvusVectorStore:
        try:
            vector_store = MilvusVectorStore(
                uri=os.getenv('CLIENT_URI'),
                token=os.getenv('CLIENT_TOKEN'),
                collection_name=collection_name,
                dim=1536,
                embedding_field='embeddings',
                enable_sparse=True,
                enable_dense=True,
                overwrite=False,  # CHANGE IT FOR DEVELOPMENT STAGE ONLY
                sparse_embedding_function=self.bm25function, #type: ignore
                search_config={"nprobe": 60},
                similarity_metric="IP",
                consistency_level="Session",
                hybrid_ranker="RRFRanker",
                hybrid_ranker_params={"k": 80}
            )
            return vector_store
        except Exception as x:
            raise x

    async def is_collection_name_exist(self, collection_name: str ,client: AsyncMilvusClient) -> bool:
        return await client.has_collection(collection_name=collection_name)

    async def alter_if_collection_name_not_exist(self,collection_name: str, client: AsyncMilvusClient) -> None:
        await client.alter_collection_properties(
            collection_name=collection_name,
            properties={"collection.ttl.seconds": 1296000}  # 15 days conversion
        )

    async def get_vector_store_chat_history(self, user_id: str) -> MilvusVectorStore:

        vector_store: Optional[MilvusVectorStore] = self.cache.get(user_id)

        if vector_store:
            return vector_store

        async with self.master_lock:
            if user_id not in self.user_locks:
                self.user_locks[user_id] = asyncio.Lock()
            specific_lock = self.user_locks[user_id]

        async with specific_lock:
            try:
                vector_store = self.cache.get(user_id)

                if vector_store:
                    return vector_store

                collection_name = self.user_id_to_collection_name(user_id)

                client = await self._milvus_client()

                existing_collection = await self.is_collection_name_exist(collection_name, client)

                new_vector_store = self._vector_store_with_bgem3(collection_name)

                if not existing_collection:
                    await self.alter_if_collection_name_not_exist(collection_name, client)

                self.cache[user_id] = new_vector_store

                return new_vector_store
            except Exception as ex:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )