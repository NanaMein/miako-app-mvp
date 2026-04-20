from sqlmodel import SQLModel
from typing import Optional

class ConversationBase(SQLModel):
    conversation_name: Optional[str]


