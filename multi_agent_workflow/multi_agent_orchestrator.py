from typing import Optional, Any
from fastapi import HTTPException, status, FastAPI
from pydantic import BaseModel
from models.message_model import MessageBase, Role, Message
from databases.database import get_session
from fastapi import Depends
from sqlmodel import select, SQLModel, delete
from sqlalchemy.ext.asyncio import AsyncSession
from multi_agent_workflow.crewai_crew.crew import MultiAgentWorkflow as AgentsWorkflow



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

@app.get("/", status_code=status.HTTP_200_OK)
def hello():
    return {"hello": "world"}

@app.post("/send-message", response_model=MessageBase, status_code=status.HTTP_201_CREATED)
def send_msg(msg: MessageBase):
    result_obj = workflow_orchestrator(inputs=msg.content)
    result = MessageBase(
        role=Role.ASSISTANT.value,
        content=result_obj
    )
    return result


if __name__ == "__main__":

    import uvicorn
    print("RUNNING UVICORN")
    uvicorn.run(
        "multi_agent_orchestrator:app",
        host="0.0.0.0",
        port=8888,
        reload=True
    )
