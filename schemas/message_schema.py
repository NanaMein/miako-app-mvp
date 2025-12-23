from typing import Optional
from datetime import datetime, timezone
from sqlmodel import SQLModel, Field
from sqlalchemy import Column, DateTime
from enum import Enum

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"

class MessageBaseSchema(SQLModel):
    role: Role = Role.USER
    content: str

# class Message(MessageBaseSchema, table=True):
#     id: Optional[int] = Field(primary_key=True, default=None)
#     date_timestamp: datetime = Field(
#         default_factory=lambda: datetime.now(timezone.utc),
#         sa_column=Column(
#             DateTime(timezone=True),
#             nullable=False
#         )
#     )