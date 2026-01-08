from typing import Optional, Any
from fastapi import HTTPException, status, FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from models.message_model import MessageBase, Role, Message
from databases.database import get_session
from fastapi import Depends
from sqlmodel import select, SQLModel, delete
from sqlalchemy.ext.asyncio import AsyncSession
from multi_agent_workflow.crewai_crew.crew import MultiAgentWorkflow as AgentsWorkflow
from datetime import datetime, timezone



agents = AgentsWorkflow()

def workflow_orchestrator(inputs: Any):
    try:
        crewai = agents.crew().kickoff(inputs=inputs)
        return crewai.raw
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unexpected Error Handling: {ex}"
        )



app = FastAPI()



async def list_all_conversations(
        session: AsyncSession,
        offset: int = 0,
        limit: int = 100
    ):
    statement = select(Message).offset(offset).limit(limit)
    result = await session.execute(statement=statement)
    messages = result.scalars().all()

    return [
        {"role": msg.role, "content":msg.content, "time_stamp":str(msg.date_timestamp) }
        for msg in messages
    ]

@app.post("/v1/send-message", response_model=MessageBase, status_code=status.HTTP_201_CREATED)
async def send_async_msg_v1(payload: MessageBase, session: AsyncSession = Depends(get_session)):

    if payload.role != Role.USER:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User should be used only")
    history = await list_all_conversations(session=session)

    crew_input = {
        "history": history,
        "user_input": payload.content
    }
    crew_output = await run_in_threadpool(workflow_orchestrator, inputs=crew_input)


    user_msg = Message(
        role=Role.USER,
        content=payload.content
    )
    session.add(user_msg)

    asst_msg = Message(
        role=Role.ASSISTANT,
        content=crew_output
    )
    session.add(asst_msg)

    await session.commit()
    await session.refresh(user_msg)
    await session.refresh(asst_msg)
    return asst_msg


@app.post("/v2/send-message", response_model=MessageBase, status_code=status.HTTP_201_CREATED)
async def send_async_msg_v1(payload: MessageBase, session: AsyncSession = Depends(get_session)):

    if payload.role != Role.USER:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User should be used only")

    history = await list_all_conversations(session=session)
    full_history = history + [
        {"role":"user", "content":payload.content, "time_stamp": str(datetime.now(timezone.utc))}
    ]

    crew_input = {
        "history": full_history,
    }
    crew_output = await run_in_threadpool(workflow_orchestrator, inputs=crew_input)

    user_msg = Message(
        role=Role.USER,
        content=payload.content
    )

    asst_msg = Message(
        role=Role.ASSISTANT,
        content=crew_output
    )

    session.add(user_msg)
    session.add(asst_msg)

    await session.commit()
    await session.refresh(user_msg)
    await session.refresh(asst_msg)

    return asst_msg


@app.delete("/clear-message", status_code=status.HTTP_204_NO_CONTENT)
async def clear_message(session: AsyncSession = Depends(get_session)):
    result = await session.execute(delete(Message))
    await session.commit()


if __name__ == "__main__":

    import uvicorn
    print("RUNNING UVICORN")
    uvicorn.run(
        "multi_agent_orchestrator:app",
        host="0.0.0.0",
        port=8888,
        reload=True
    )
