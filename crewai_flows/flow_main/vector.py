from typing import Optional
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.milvus.utils import  BM25BuiltInFunction
from dotenv import load_dotenv
from cachetools import TTLCache
from pymilvus import MilvusException, MilvusClient
from fastapi import HTTPException, status
import grpc
import os
from grpc.aio import AioRpcError
from collections import defaultdict
from threading import Lock

load_dotenv()



class MilvusVectorStoreClass:
    def __init__(self):
        self.cache = TTLCache(maxsize=100, ttl=250)
        self.master_lock = Lock()
        self.user_locks = defaultdict(Lock)
        # self.collection_name: str = f"Collection_Name_{self.user_id.strip()}_2025"
        # self.name: Optional[str] = None

    def _vector_store(self, collection_name: str):
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

    def _milvus_client(self) -> MilvusClient:
        client = MilvusClient(
            uri=os.getenv('CLIENT_URI'),
            token=os.getenv('CLIENT_TOKEN')
        )
        return client

    def user_id_to_collection_name(self, user_id: str):
        # stripped_user_id = user_id.strip()
        len_of_16_str = user_id.strip()[:16]
        return f"Collection_Of_{len_of_16_str}_2025_2026"

    def is_collection_name_exist(self, collection_name: str ,client: MilvusClient) -> bool:
        return client.has_collection( collection_name=collection_name )

    def alter_if_collection_name_not_exist(self,collection_name: str, client: MilvusClient):
        client.alter_collection_properties(
            collection_name=collection_name,
            properties={"collection.ttl.seconds": 1296000}  # 15 days conversion
        )

    def get_vector_chat_history(self, user_id: str):
        existing_cache = self.cache.get(user_id)
        if existing_cache is not None:
            return existing_cache

        with self.master_lock:
            specific_lock = self.user_locks[user_id]

        with specific_lock:
            existing_cache = self.cache.get(user_id)

            if existing_cache is not None:
                return existing_cache

            try:

                collection_name = self.user_id_to_collection_name(user_id=user_id)

                client = self._milvus_client()

                existing_collection = self.is_collection_name_exist(
                    collection_name=collection_name,
                    client=client
                )

                vector_store = self._vector_store(collection_name=collection_name)

                if not existing_collection:
                    self.alter_if_collection_name_not_exist(
                        collection_name=collection_name,
                        client=client
                    )

                self.cache.expire()
                self.cache[user_id] = vector_store
                return self.cache[user_id]
            except (MilvusException, AioRpcError) as ma:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Error in Milvus: {ma}"
                )

            except Exception as ex:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Unexpected Error: {ex}"
                )


    def get_vector_chat_history_original(self, user_id: str):
        existing_cache = self.cache.get(user_id)
        if existing_cache is not None:
            return existing_cache

        # if user_id in self._cache:
        #     return self._cache[user_id]

        with self.master_lock:
            specific_lock = self.user_locks[user_id]

        with specific_lock:
            existing_cache = self.cache.get(user_id)

            if existing_cache is not None:
                return existing_cache
            try:
                collection_name: str = f"Collection_Of_{user_id.strip()}_2025_2026"

                client = MilvusClient(
                    uri=os.getenv('CLIENT_URI'),
                    token=os.getenv('CLIENT_TOKEN')
                )
                existing_collection = client.has_collection(
                    collection_name=collection_name
                )
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

                if not existing_collection:
                    client.alter_collection_properties(
                        collection_name=collection_name,
                        properties={"collection.ttl.seconds": 1296000} #15 days conversion
                    )

                # if user_id in self._cache:
                #     return self._cache[user_id]
                #####

                ##some data
                self.cache.expire()
                self.cache[user_id] = vector_store
                return self.cache[user_id]
            except Exception as ex:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error in cache vector: {ex}"
                )