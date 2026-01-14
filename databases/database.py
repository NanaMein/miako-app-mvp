import os
from dotenv import load_dotenv
from core.config import settings
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

# DATABASE_URL=os.getenv("DATABASE_URL")
DATABASE_URL=settings.DATABASE_URL.get_secret_value()

engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True
)

async_session_maker = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session