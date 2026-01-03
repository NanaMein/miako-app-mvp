from typing import Optional, TYPE_CHECKING
from datetime import datetime, timezone
from sqlmodel import Field, Relationship
from sqlalchemy import Column, DateTime
from schemas.message_schema import MessageBaseSchema

if TYPE_CHECKING:
    from models.conversation_model import Conversation


class Message(MessageBaseSchema, table=True):

    id: Optional[int] = Field(primary_key=True, default=None)

    date_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False
        )
    )
    conversation_id: Optional[int] = Field(nullable=False, foreign_key="conversation.id")
    conversation: Optional["Conversation"] = Relationship(back_populates="messages")
