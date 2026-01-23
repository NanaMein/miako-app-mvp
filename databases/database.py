from core.config import settings
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker


class DatabaseManager:
    def __init__(self):
        self.engine = create_async_engine(
            settings.DATABASE_URL,
            echo=True,
            future=True,
            pool_size = 5,  # Connection pool size
            max_overflow = 10,  # Extra connections when busy
            pool_pre_ping = True,  # Check connections before using them
        )

        self.async_session_maker = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

dbm = DatabaseManager()

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with dbm.async_session_maker() as session:
        yield session

