# tests/conftest.py
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

# Import your models here so SQLModel knows about them before creating tables
from models.user_model import User
from models.conversation_model import Conversation
from models.message_model import Message

# Using in-memory SQLite for speed and isolation
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """
    Creates an instance of the default event loop for the test session.
    Required for async fixtures with 'session' scope.
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_engine():
    """
    Creates the database engine once for the whole test session.
    """
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    yield engine

    # Clean up
    await engine.dispose()


@pytest.fixture
async def session(async_engine):
    """
    Creates a new session for each test function.
    Rollbacks or closes happen automatically when the context ends.
    """
    async_session_maker = sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )

    async with async_session_maker() as s:
        yield s