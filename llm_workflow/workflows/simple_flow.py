from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import delete
from models.message_model import Message
from schemas.message_schema import MessageBaseSchema, Role
from databases.database import get_session
from llm_workflow.workflows.flow_chatbot import FlowMainWorkflow
from fastapi import FastAPI, HTTPException, status, Depends
from llm_workflow.workflows.main_workflow import MainFlowStates, MainWorkflow

flow_ = MainWorkflow()
app = FastAPI()





@app.delete("/clear-message", status_code=status.HTTP_204_NO_CONTENT)
async def clear_message(session: AsyncSession = Depends(get_session)):
    await session.execute(delete(Message))
    await session.commit()


async def inputs_test(**kwargs):
    return await flow_.kickoff_async(inputs={**kwargs})

class FlowService:
    def __init__(self, **kwargs):
        self.service=flow_.kickoff_async(inputs={**kwargs})

class PayloadValidation(BaseModel):
    message: str


@app.post("/flow-main-send", response_model=PayloadValidation)
async def flow_send(payload: PayloadValidation, session: AsyncSession = Depends(get_session)):
    try:

        flow_serve = FlowService(
            input_user_id="test",
            input_message=payload.message,
            async_session=session
        )
        service = await flow_serve.service
        return PayloadValidation(message=service)
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Error error error: {ex}"
        )

if __name__ == "__main__":


    import uvicorn
    print("RUNNING UVICORN")
    uvicorn.run(
        "simple_flow:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


