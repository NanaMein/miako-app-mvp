from sqlmodel import Field, Relationship
from typing import Optional, TYPE_CHECKING
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime
from schemas.user_schema import UserBase
from uuid import uuid4, UUID

if TYPE_CHECKING:
    from models.conversation_model import Conversation


class User(UserBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    user_name: Optional[str]

    hashed_password: str

    uuid: UUID = Field(default_factory=uuid4, index=True)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False
        )
    )

    conversations: list["Conversation"] = Relationship(back_populates="user")


