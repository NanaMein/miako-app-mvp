from typing import Optional
from datetime import datetime, timezone
from sqlmodel import Field
from sqlalchemy import Column, DateTime
from schemas.message_schema import MessageBaseSchema



class Message(MessageBaseSchema, table=True):

    id: Optional[int] = Field(primary_key=True, default=None)

    date_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(
            DateTime(timezone=True),
            nullable=False
        )
    )
