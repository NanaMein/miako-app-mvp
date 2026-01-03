from sqlmodel import Field, Relationship
from typing import Optional, TYPE_CHECKING
from datetime import datetime, timezone
from sqlalchemy import Column, DateTime
from schemas.conversation_schema import ConversationBase


if TYPE_CHECKING:
    from models.message_model import Message
    from models.user_model import User

class Conversation(ConversationBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False
        )
    )
    user_id: Optional[int] = Field(nullable=False, foreign_key="user.id")
    user: Optional["User"] = Relationship(back_populates="conversations")
    messages: list["Message"] = Relationship(back_populates="conversation")
