from typing import Optional
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import  BM25BuiltInFunction, BGEM3SparseEmbeddingFunction
from dotenv import load_dotenv
from cachetools import TTLCache, LRUCache
from pymilvus import MilvusException, MilvusClient, AsyncMilvusClient
from fastapi import HTTPException, status
import grpc
import os
from grpc.aio import AioRpcError
from collections import defaultdict
from threading import Lock
import time
import asyncio
from sample_logger import logger
load_dotenv()

class MilvusVectorStoreClassAsync:

    def __init__(self):
        self.cache = LRUCache(maxsize=100)
        self.master_lock = asyncio.Lock()
        self.user_locks = {}
        self._bgem3function = None

    @property
    def bgem3function(self) -> BGEM3SparseEmbeddingFunction:
        if self._bgem3function is None:
            logger.info("Starting to instantiate BGEM3SparseEmbeddingFunction...")
            self._bgem3function = BGEM3SparseEmbeddingFunction()
            logger.success("BGEM3SparseEmbeddingFunction is instantiated!")
        return self._bgem3function

    def user_id_to_collection_name(self, user_id: str) -> str:
        len_of_16_str = user_id.strip()[:16]
        return f"Collection_Of_{len_of_16_str}_2025_2026"

    async def _milvus_client(self) -> AsyncMilvusClient:
        client = AsyncMilvusClient(
            uri=os.getenv('CLIENT_URI'),
            token=os.getenv('CLIENT_TOKEN')
        )
        logger.success("Successfully make milvus client connected to server")
        return client

    def _vector_store_with_bm25(self, collection_name: str) -> MilvusVectorStore:
        try:
            bm25_function = BM25BuiltInFunction(
                analyzer_params={
                    "tokenizer": "multilingual",  # Try 'multilingual' for better script handling
                    "filter": [
                        "lowercase",
                        {"type": "length", "max": 40},
                    ],
                },
                enable_match=True,
                input_field_names=["text"],  # The source field for BM25
                output_field_names=["sparse_embeddings"]  # Where the sparse vector goes
            )

            vector_store = MilvusVectorStore(
                uri=os.getenv('CLIENT_URI'),
                token=os.getenv('CLIENT_TOKEN'),
                collection_name=collection_name,
                dim=1536,
                embedding_field='embeddings',
                enable_sparse=True,
                enable_dense=True,
                overwrite=False,  # CHANGE IT FOR DEVELOPMENT STAGE ONLY
                # sparse_embedding_function=BGEM3SparseEmbeddingFunction(),
                sparse_embedding_function=bm25_function,
                search_config={"nprobe": 60},
                similarity_metric="IP",
                consistency_level="Session",
                hybrid_ranker="RRFRanker",
                hybrid_ranker_params={"k": 80},
            )
            return vector_store
        except Exception as x:
            raise x

    def _vector_store_with_bgem3(self, collection_name: str) -> MilvusVectorStore:
        try:
            logger.info("Starting to make connection with Zilliz Cloud vector")
            vector_store = MilvusVectorStore(
                uri=os.getenv('CLIENT_URI'),
                token=os.getenv('CLIENT_TOKEN'),
                collection_name=collection_name,
                dim=1536,
                embedding_field='embeddings',
                enable_sparse=True,
                enable_dense=True,
                overwrite=False,  # CHANGE IT FOR DEVELOPMENT STAGE ONLY
                sparse_embedding_function=self.bgem3function,
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
        logger.info("Checking to see if Zilliz cloud has that collection name")
        return await client.has_collection(collection_name=collection_name)

    async def alter_if_collection_name_not_exist(self,collection_name: str, client: AsyncMilvusClient) -> None:
        await client.alter_collection_properties(
            collection_name=collection_name,
            properties={"collection.ttl.seconds": 1296000}  # 15 days conversion
        )

    async def get_vector_store_chat_history(self, user_id: str) -> MilvusVectorStore:
        user_logger = logger.bind(user=user_id)
        user_logger.trace("Get vector store by user id")

        vector_store: Optional[MilvusVectorStore] = self.cache.get(user_id)

        if vector_store:
            user_logger.success("Fast cache hit for vector store")
            return vector_store

        async with self.master_lock:
            if user_id not in self.user_locks:
                self.user_locks[user_id] = asyncio.Lock()
            specific_lock = self.user_locks[user_id]

        async with specific_lock:
            try:
                user_logger.debug("Initiating Vector connection in Lock...")
                vector_store = self.cache.get(user_id)

                if vector_store:
                    user_logger.success("Success vector connection the second time")
                    return vector_store

                user_logger.debug("Started to make new connection to vector and cache it...")
                collection_name = self.user_id_to_collection_name(user_id)
                user_logger.info(f"Collection name [{collection_name}] created")

                client = await self._milvus_client()

                existing_collection = await self.is_collection_name_exist(collection_name, client)

                new_vector_store = self._vector_store_with_bgem3(collection_name)

                if not existing_collection:
                    await self.alter_if_collection_name_not_exist(collection_name, client)
                    user_logger.success(f"Altered properties for [{collection_name}] for the first time for TTL.")

                self.cache[user_id] = new_vector_store

                user_logger.success("Successfully cached vector store in lock.")
                return new_vector_store
            except Exception as ex:
                user_logger.exception(f"Zilliz vector store error: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error"
                )