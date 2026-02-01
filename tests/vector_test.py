import pytest
import pytest_asyncio
from llama_index.vector_stores.milvus import MilvusVectorStore
from llm_workflow.vector_stores.vector_memory_store import ConversationMemoryStore
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


@pytest.fixture(scope="session")
def memory_store(user_id):
    memory = ConversationMemoryStore(user_id=user_id)
    return memory

@pytest_asyncio.fixture(scope="session")
async def query_memory(memory_store: ConversationMemoryStore):
    mem = await memory_store.add_(
        user_message="hello",assistant_message="Hello there how are you"
    )
    return mem

@pytest.mark.asyncio
async def test_query_memory_saved(query_memory):
    print("\nhello world")
    assert query_memory is not False

@pytest_asyncio.fixture(scope="session")
async def test_memory_by_query(memory_store):
    hello_content = await memory_store.show_(query="hello")
    print(hello_content)
    return hello_content

@pytest.mark.asyncio
async def test_memory_if_not_none(test_memory_by_query):
    assert test_memory_by_query is not ""

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))