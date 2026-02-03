import pytest
import pytest_asyncio
import asyncio
import uuid
from llama_index.vector_stores.milvus import MilvusVectorStore
from llm_workflow.vector_stores.vector_memory_store import ConversationMemoryStore
from llm_workflow.vector_stores.vector_connection import MilvusVectorStoreConnection, milvus_client


# -------------------------------------------------------------------------
# FIXTURES
# -------------------------------------------------------------------------

@pytest.fixture
def user_id():
    """Generates a unique user ID for every test to avoid collision."""
    return f"user_{uuid.uuid4().hex}"


@pytest_asyncio.fixture
async def memory_store_object(user_id):
    """Creates a memory store with default TTL."""
    cms = ConversationMemoryStore(
        user_id=user_id,
        ttl_hours=0.5,
        ttl_mins=0
    )
    yield cms
    # Cleanup logic could go here, but we do it in cleanup_collection


@pytest_asyncio.fixture
async def cleanup_collection(user_id):
    """
    Survivalist Tool: Automatically drops the collection after test.
    Keeps the Milvus instance clean.
    """
    yield
    try:
        conn = MilvusVectorStoreConnection(user_id=user_id)
        client = await milvus_client()
        if await client.has_collection(conn.collection_name):
            await client.drop_collection(conn.collection_name)
            print(f"\n[CleanUp] Dropped collection: {conn.collection_name}")
    except Exception as e:
        print(f"[CleanUp] Warning: Failed to drop collection: {e}")


# -------------------------------------------------------------------------
# TESTS
# -------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collection_name_generation(user_id):
    """Test if collection name cleans special characters correctly."""
    messy_id = user_id + "!@#$%"
    obj = MilvusVectorStoreConnection(user_id=messy_id)
    assert "!@#$%" not in obj.collection_name
    assert "Collection_Of_" in obj.collection_name


@pytest.mark.asyncio
async def test_add_and_query_memory(memory_store_object, cleanup_collection):
    """Test adding memory and retrieving it."""

    # 1. Add Memory
    added = await memory_store_object.add_(
        user_message="Hello Nanami",
        assistant_message="Hello User-san!"
    )
    assert added is True

    # 2. Verify Vector Store Type
    # Accessing internal milvus_store to check connection
    vs = await memory_store_object.milvus_store.get_vector_store()
    assert isinstance(vs, MilvusVectorStore)

    # 3. Retrieve Memory
    # We query 'Nanami' which should match the context added above
    result = await memory_store_object.show_("Nanami")
    print(f"\nQuery Result: {result}")

    assert result is not None
    assert "Hello User-san!" in result or "Hello Nanami" in result


@pytest.mark.asyncio
async def test_ttl_calculation_logic(user_id):
    """Test if hours/mins convert to seconds correctly."""
    # 1 Hour + 30 Mins = 3600 + 1800 = 5400 seconds
    conn = MilvusVectorStoreConnection(user_id=user_id, default_ttl_hours=1, default_ttl_mins=30)
    assert conn.default_ttl == 5400

    # 0 Hour + 0 Mins = 0 (No TTL)
    conn_zero = MilvusVectorStoreConnection(user_id=user_id)
    assert conn_zero.default_ttl == 0


@pytest.mark.asyncio
async def test_ttl_enforcement_in_milvus(user_id, cleanup_collection):
    """
    Integration Test:
    1. Create collection with specific TTL.
    2. Check if Milvus actually applied that property.
    """
    ttl_hours = 1
    ttl_seconds = 3600

    memory = ConversationMemoryStore(
        user_id=user_id,
        ttl_hours=ttl_hours,
        ttl_mins=0
    )

    # Trigger connection creation
    await memory.add_(user_message="Test TTL", assistant_message="Checking TTL")

    # Check Property directly from Milvus Client
    actual_ttl = await memory.milvus_store._check_client_property_ttl()

    print(f"\nExpected TTL: {ttl_seconds}, Actual TTL in Milvus: {actual_ttl}")
    assert actual_ttl == ttl_seconds


@pytest.mark.asyncio
async def test_dynamic_ttl_update(user_id, cleanup_collection):
    """
    Advanced Test:
    1. Create store with TTL A.
    2. Create NEW store instance for SAME user but with TTL B.
    3. Ensure the system updates the Milvus collection property dynamically.
    """
    # Step 1: Initialize with 1 hour TTL
    mem_v1 = ConversationMemoryStore(user_id=user_id, ttl_hours=1, ttl_mins=0)
    await mem_v1.add_(user_message="Init", assistant_message="Init")

    ttl_v1 = await mem_v1.milvus_store._check_client_property_ttl()
    assert ttl_v1 == 3600

    # Step 2: Initialize with 2 hours TTL (Same User)
    mem_v2 = ConversationMemoryStore(user_id=user_id, ttl_hours=2, ttl_mins=0)

    # Trigger the logic (get_vector_store triggers _should_alter_properties)
    await mem_v2.milvus_store.get_vector_store()

    # Step 3: Verify Milvus was updated
    ttl_v2 = await mem_v2.milvus_store._check_client_property_ttl()

    print(f"\nTTL V1: {ttl_v1} -> TTL V2: {ttl_v2}")
    assert ttl_v2 == 7200  # 2 hours * 3600


@pytest.mark.asyncio
async def test_concurrency_stress(user_id, cleanup_collection):
    """
    Survival Test:
    Simulate multiple async calls to ensure locks work and don't deadlock.
    """
    memory = ConversationMemoryStore(user_id=user_id)

    async def add_data(idx):
        return await memory.add_(
            user_message=f"Msg {idx}",
            assistant_message=f"Resp {idx}"
        )

    # Run 5 adds concurrently
    tasks = [add_data(i) for i in range(5)]
    results = await asyncio.gather(*tasks)

    assert all(results)  # All should be True

    # Verify data count (approximated via show_)
    context = await memory.show_("Msg")
    assert context is not None



if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))