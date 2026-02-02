import pytest
import pytest_asyncio
import uuid
from llama_index.vector_stores.milvus import MilvusVectorStore
from llm_workflow.vector_stores.vector_memory_store import ConversationMemoryStore
from llm_workflow.vector_stores.vector_connection import MilvusVectorStoreConnection, milvus_client_v1




@pytest.fixture
def user_id():
    # uid = str(uuid.uuid4())
    uid = "Happy Sex"
    uid = "test property"
    return uid

@pytest.fixture
def collection_name(user_id):
    return user_id.replace("-", "").replace(" ", "").replace("_", "")
###````````````````````````````````````````````````````````````````````````````````````````

@pytest_asyncio.fixture
async def collection_name_for_milvus(collection_name):
    obj = MilvusVectorStoreConnection(collection_name)
    return obj.collection_name

@pytest_asyncio.fixture
async def memory_store_object(collection_name):
    cms = ConversationMemoryStore(
        user_id=collection_name,
        ttl_hours=.5, ttl_mins=60
    )
    yield cms

@pytest_asyncio.fixture
async def vector_object(collection_name):
    mvsc = MilvusVectorStoreConnection(collection_name)
    yield mvsc

@pytest.mark.asyncio
async def test_add_and_query(memory_store_object, vector_object):
    added = await memory_store_object.add_(
        user_message="Hello again",
        assistant_message="Oh, how are you doing?"
    )
    assert added is True

    vs = await vector_object.get_vector_store()
    assert isinstance(vs, MilvusVectorStore)

    result = await memory_store_object.show_("Hello")
    print("Result from show",result)
    assert result is not None


@pytest.mark.asyncio
async def test_check_ttl_property(vector_object, collection_name_for_milvus):

    client = await milvus_client_v1()
    props = await client.describe_collection(collection_name=collection_name_for_milvus)

    properties = props.get("properties", {})
    prop = properties.get("collection.ttl.seconds")
    # if isinstance(prop, str) and prop.isdigit():
    #
    #     prop = int(prop)
    assert prop is None or isinstance(prop, int) or (isinstance(prop, str) and prop.isdigit())

    if isinstance(prop, str) and prop.isdigit():
        prop = int(prop)
    print("TTL seconds:", prop)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))