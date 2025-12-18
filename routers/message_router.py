from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from models.message_model import Message, MessageBase
from sqlmodel import select
from databases.database import get_session

router = APIRouter(
    prefix="/message",
    tags=["/message/"]
)


@router.post("/create-new", response_model=Message, status_code=status.HTTP_201_CREATED)
async def create_new_message(payload: MessageBase, session: AsyncSession = Depends(get_session)):
    message = Message(
        role=payload.role,
        content=payload.content
    )
    session.add(message)
    await session.commit()
    await session.refresh(message)
    return message

@router.get("/get-all", response_model=list[Message], status_code=status.HTTP_200_OK)
async def get_all_messages(session: AsyncSession = Depends(get_session), offset: int = 0, limit: int = 100):
    statement = select(Message).offset(offset=offset).limit(limit=limit)
    result = await session.execute(statement=statement)
    messages = result.scalars().all()
    return messages