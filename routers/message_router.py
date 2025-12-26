from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from models.message_model import Message
from schemas.message_schema import MessageBaseSchema, Role
from multi_agent_workflow.multi_agent_orchestrator import workflow_orchestrator
from sqlmodel import select
from databases.database import get_session

router = APIRouter(
    prefix="/message",
    tags=["/message/"]
)


@router.post("/create-new", response_model=Message, status_code=status.HTTP_201_CREATED)
async def create_new_message(payload: MessageBaseSchema, session: AsyncSession = Depends(get_session)):
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


async def get_list_conversation_list(
        session: AsyncSession,
        offset: int = 0,
        limit: int = 100
):
    try:
        statement = select(Message).offset(offset).limit(limit)
        result = await session.execute(statement)
        messages = result.scalars().all()
        return [
            {"role":msg.role, "content":msg.content, "date_timestamp":str(msg.date_timestamp)}
            for msg in messages
        ]
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error in List[Message]: {ex}")


@router.post("/send-message", response_model=MessageBaseSchema, status_code=status.HTTP_201_CREATED)
async def send_message(payload: MessageBaseSchema,session: AsyncSession = Depends(get_session)):

    if payload.role != "user":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User is required")

    try:
        history = await get_list_conversation_list(session=session)

        user_message = Message(
            role=Role.USER,
            content=payload.content
        )
        full_history = history + [user_message]

        output_message = workflow_orchestrator(inputs=full_history)

        asst_msg = Message(
            role=Role.ASSISTANT,
            content=output_message
        )
        session.add(user_message)
        session.add(output_message)
        await session.commit()
        await session.refresh(user_message)
        await session.refresh(asst_msg)
        return asst_msg
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail=f"Error POST: {ex}"
        )

from sqlmodel import delete
@router.delete("/clear", status_code=status.HTTP_204_NO_CONTENT)
async def delete_list(session: AsyncSession = Depends(get_session)):
    result = await session.execute(delete(Message))
    await session.commit()

