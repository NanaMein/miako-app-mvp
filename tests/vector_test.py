import pytest
import pytest_asyncio
from llama_index.vector_stores.milvus import MilvusVectorStore

from llm_workflow.vector_stores.vector_connection import MilvusVectorStoreConnection

@pytest.fixture(scope="session")
def user_id():
    return "feb_1_26"

@pytest.fixture(scope="session")
def vector_connection(user_id):
    return MilvusVectorStoreConnection(user_id=user_id)

@pytest_asyncio.fixture(scope="session")
async def vector_store(vector_connection):
    return await vector_connection.get_vector_store()

@pytest.mark.asyncio
async def test_vector_store_if_not_none(vector_store):
    assert vector_store is not None

@pytest.mark.asyncio
async def test_vector_store_type(vector_store):
    assert isinstance(vector_store, MilvusVectorStore)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))