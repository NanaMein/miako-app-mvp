from typing import Optional, Union, Any
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import  BM25BuiltInFunction
from dotenv import load_dotenv
from cachetools import LRUCache
from pymilvus import AsyncMilvusClient
from fastapi import HTTPException, status
import os
import asyncio
from llm_workflow.config_files.config import settings_for_workflow
from llm_workflow.config_files.locking import LockManager


CLIENT_URI=settings_for_workflow.CLIENT_URI.get_secret_value()
CLIENT_TOKEN=settings_for_workflow.CLIENT_TOKEN.get_secret_value()
BM25FUNCTION = BM25BuiltInFunction()
VECTOR_CACHE = LRUCache(maxsize=1000)
CACHE_FOR_LOCK = LRUCache(maxsize=1000)
ASYNC_LOCK = asyncio.Lock()
CLIENT = AsyncMilvusClient(
    uri=CLIENT_URI,
    token=CLIENT_TOKEN
)


class MilvusVectorStoreConnection:

    def __init__(self, user_id: Union[str, Any], default_ttl_hours: float = 0, default_ttl_mins: int = 1):
        self._user_id = user_id
        self._default_ttl_hours = default_ttl_hours
        self._default_ttl_min = default_ttl_mins
        self._lock = LockManager(
            user_id=self._user_id,
            asyncio_lock=ASYNC_LOCK,
            cache=CACHE_FOR_LOCK
        )


    @property
    def bm25function(self) -> BM25BuiltInFunction:
        return BM25FUNCTION


    @property
    def collection_name(self) -> str:
        len_of_16_str = self._user_id.strip()[:20]
        return f"Collection_Of_{len_of_16_str}_2025_2026"


    @property
    def vector_cache(self):
        return VECTOR_CACHE


    @property
    def default_ttl(self):
        hours = int(self._default_ttl_hours * 3600)
        mins = self._default_ttl_min * 60
        return hours + mins


    def _vector_store_with_bm25(self) -> MilvusVectorStore:
        try:
            vector_store = MilvusVectorStore(
                uri=CLIENT_URI,
                token=CLIENT_TOKEN,
                collection_name=self.collection_name,
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
                hybrid_ranker_params={"k": 60}
            )
            return vector_store
        except Exception as x:
            raise x


    async def _is_collection_name_exist(self) -> bool:
        return await CLIENT.has_collection(collection_name=self.collection_name)


    async def _alter_if_collection_name_not_exist(self) -> None:
        await CLIENT.alter_collection_properties(
            collection_name=self.collection_name,
            properties={"collection.ttl.seconds": self.default_ttl}
        )


    async def _core_vector_store_logic(self) -> MilvusVectorStore:
        if self._user_id in self.vector_cache:
            return self.vector_cache[self._user_id]

        lock = await self._lock.get_lock()
        async with lock:
            try:
                if self._user_id in self.vector_cache:
                    return self.vector_cache[self._user_id]

                vector_store = self.vector_cache.get(self._user_id)
                if vector_store:
                    return vector_store

                existing_collection = await self._is_collection_name_exist()

                new_vector_connection: MilvusVectorStore = self._vector_store_with_bm25()

                if not existing_collection:
                    await self._alter_if_collection_name_not_exist()

                self.vector_cache[self._user_id] = new_vector_connection

                return new_vector_connection

            except Exception as ex:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Internal server error: {ex}"
                )

    async def _reconnection_and_retry_logic(self) -> MilvusVectorStore:

        try:
            vector = await self._core_vector_store_logic()
            return vector
        except HTTPException:
            pass

        try:
            self.vector_cache.pop(self._user_id)
            vector = await self._core_vector_store_logic()
            return vector
        except HTTPException as ex:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal error: {ex}"
            )

    async def get_vector_store(self) -> MilvusVectorStore:
        return await self._reconnection_and_retry_logic()




load_dotenv()

class MilvusVectorStoreClassAsyncOriginal:

    def __init__(self):
        self.cache = LRUCache(maxsize=100)
        self.master_lock = asyncio.Lock()
        self.user_locks = {}


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
                hybrid_ranker_params={"k": 60}
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