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
from llm_workflow.chat_completions.groq_llm import ChatCompletionsClass
from llm_workflow.workflows.main_workflow import MainFlowStates, LLMWorkflow, flow_kickoff

flow_ = LLMWorkflow()
app = FastAPI()




ROLE = {
    "system": {"role": "system"},
    "user":{"role" : "user"},
    "assistant": {"role": "assistant"}
}
def make_message(role:str, content:str):
    prompt = ROLE.get(role).copy()
    prompt["content"] = content
    return prompt



SAMPLE_CONVERSATION= [
    make_message("user","Kamusta? May problema po ako sa aking laptop. Hindi siya magsisimula."),

    make_message("assistant"," Magandang araw po! Pasensya na po, ano pong exact na nangyayari sa inyong laptop? May error message po ba na lumalabas?"),

    make_message("user","Hindi po, wala pong error message. Parang dead lang talaga siya. Sinubukan ko nang i-plug sa charger pero walang reaction."),

    make_message("assistant","Naiintindihan ko po ang inyong problema. Pwede po bang tanungin kung gaano na po katagal 'to? At nangyari po ba ito bigla o may nanguna pong incident?"),

    make_message("user","Nangyari po ito kahapon pagkatapos kong mag-update ng Windows. Nag-shutdown siya ng normal pero ngayon hindi na siya bumubukas."),

    make_message("assistant"," Ah, salamat po sa impormasyon! Malamang po na may issue sa Windows update. Pwede po ba kayong subukan ang hard reset? Pindutin lang po ang power button ng 15 seconds, pagkatapos ay pakitanggal ang charger at battery kung maaari."),

    make_message("user","Sinubukan ko na po 'yan pero hindi pa rin gumagana. Ano pong next step?"),

    make_message("assistant","Pasensya na po, Master. Pwede po ba tayong subukan ang external monitor? Baka po kasi ang issue ay sa display card o screen lamang. May VGA o HDMI port po ba ang inyong laptop?"),

    make_message("user","Wala po akong external monitor. May iba pong paraan? Baka po ba ito hardware issue?")
]
# print(SAMPLE_CONVERSATION)

chat_comp = ChatCompletionsClass()



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


@app.get("/sample-conv")
async def flow_send():
    chat_comp.cached_messages=SAMPLE_CONVERSATION
    return await chat_comp.groq_scout()


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

@app.post("/v1/flow-main-send", response_model=PayloadValidation)
async def flow_send(payload: PayloadValidation, session: AsyncSession = Depends(get_session)):
    try:
        flow_object = await flow_kickoff(
            input_user_id="test",
            input_message=payload.message,
            async_session=session
        )
        return PayloadValidation(message=flow_object)
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


