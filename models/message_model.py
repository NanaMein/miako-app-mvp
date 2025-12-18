from typing import Optional
from enum import Enum
from pydantic import BaseModel
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class MessageBase(BaseModel):
    role: Role = Role.USER
    content: str

class Message(MessageBase, SQLModel, table=True):

    id: Optional[int] = Field(primary_key=True, default=None)

    date_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": "now()"}
    )
