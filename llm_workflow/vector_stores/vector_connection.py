from typing import Union, Any
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import  BM25BuiltInFunction
from weakref import WeakKeyDictionary
from cachetools import LRUCache
from pymilvus import AsyncMilvusClient
from fastapi import HTTPException, status
from llm_workflow.config_files.config import workflow_settings
from llm_workflow.config_files.locking import LockManager
import asyncio
import re


CLIENT_URI=workflow_settings.CLIENT_URI.get_secret_value()
CLIENT_TOKEN=workflow_settings.CLIENT_TOKEN.get_secret_value()
BM25FUNCTION = BM25BuiltInFunction(
    analyzer_params={
        "tokenizer": "icu",
        "filter": [
            "lowercase",
            {"type": "length", "max": 40},
        ],
    },
    enable_match=True,
    input_field_names=["text"],
    output_field_names=["sparse_embeddings"]
)
VECTOR_CACHE = LRUCache(maxsize=1000)
CACHE_FOR_LOCK = LRUCache(maxsize=1000)
ASYNC_LOCK = asyncio.Lock()

_ASYNC_MILVUS_CLIENT = WeakKeyDictionary()
_MILVUS_LOCK = asyncio.Lock()

async def milvus_client():
    loop = asyncio.get_running_loop()

    async with _MILVUS_LOCK:
        _client = _ASYNC_MILVUS_CLIENT.get(loop)
        if _client is None:
            _client = AsyncMilvusClient(
                uri=CLIENT_URI,
                token=CLIENT_TOKEN
            )
            _ASYNC_MILVUS_CLIENT[loop] = _client
        return _client


class MilvusVectorStoreConnection:

    def __init__(self, user_id: Union[str, Any], default_ttl_hours: float = 0, default_ttl_mins: float = 0):
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
        corrected_id = re.sub(r"[^a-zA-Z0-9_]", "_", self._user_id)
        return f"Collection_Of_{corrected_id}_2025_2026"


    @property
    def vector_cache(self):
        return VECTOR_CACHE


    @property
    def default_ttl(self):
        return int(self._default_ttl_hours * 3600) + int(self._default_ttl_min * 60)


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
                sparse_embedding_field="sparse_embeddings",
                search_config={"nprobe": 60},
                similarity_metric="IP",
                consistency_level="Session",
                hybrid_ranker="RRFRanker",
                hybrid_ranker_params={"k": 60}
            )
            return vector_store
        except Exception as x:
            raise x


    async def _collection_exist(self) -> bool:
        client = await milvus_client()
        return await client.has_collection(collection_name=self.collection_name)


    async def _set_ttl_if_first_creation(self) -> None:
        client = await milvus_client()
        await client.alter_collection_properties(
            collection_name=self.collection_name,
            properties={"collection.ttl.seconds": self.default_ttl}
        )


    async def _core_vector_store_logic(self) -> MilvusVectorStore:
        if self._user_id in self.vector_cache:
            return self.vector_cache[self._user_id]

        lock = await self._lock.get_lock()
        async with lock:
            try:
                vector_store = self.vector_cache.get(self._user_id)
                if isinstance(vector_store, MilvusVectorStore):
                    return vector_store

                collection = await self._collection_exist()

                new_vector_connection = self._vector_store_with_bm25()

                if not collection:
                    await self._set_ttl_if_first_creation()

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
            self.vector_cache.pop(self._user_id, None)
            vector = await self._core_vector_store_logic()
            return vector
        except HTTPException as ex:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal error: {ex}"
            )

    async def get_vector_store(self) -> MilvusVectorStore:
        return await self._reconnection_and_retry_logic()
