import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import select, desc, delete
from models.message_model import Message
from schemas.message_schema import MessageBaseSchema, Role
from databases.database import get_session
from crewai_flows.flow_main.flow_chatbot import FlowMainWorkflow
from fastapi import FastAPI, HTTPException, status, Depends

load_dotenv()
flow_start = FlowMainWorkflow()
app = FastAPI()



@app.post("/flow-send", response_model=MessageBaseSchema)
async def flow_send(payload: MessageBaseSchema, session: AsyncSession = Depends(get_session)):
    try:
        payload_input = {
            "input_message":payload.content,
            "input_user_id":payload.role,
            "async_session": session
        }
        output_flow = await flow_start.kickoff_async(inputs=payload_input)
        assistant_message = {"role":Role.ASSISTANT.value,"content":output_flow}
        return assistant_message
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Error error error: {ex}"
        )


@app.delete("/clear-message", status_code=status.HTTP_204_NO_CONTENT)
async def clear_message(session: AsyncSession = Depends(get_session)):
    await session.execute(delete(Message))
    await session.commit()


if __name__ == "__main__":

    import uvicorn
    print("RUNNING UVICORN")
    uvicorn.run(
        "flow_chatbot:app",
        host="0.0.0.0",
        port=8888,
        reload=True
    )
