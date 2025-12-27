import os
from dotenv import load_dotenv
from typing import Optional, Any
from crewai.flow.flow import Flow, listen, start
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from groq import Groq
from groq.types.chat import (
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletion
)
from fastapi import FastAPI, HTTPException, status

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class StatesTemporarily(BaseModel):
    input_message: str = ""
    experimental_input_file: Any = None

class ChatbotStates(BaseModel):
    id: Optional[int] = Field(description="The identifier that separates from other id", default=None)
    input_message: str = Field(description="User\'s input message", default="")
    experimental_input_file: Optional[ChatCompletionUserMessageParam] = Field(description="Experimental object carrier", default=None)

class ChatbotWorkflow(Flow[ChatbotStates]):

    @start()
    def start_with_user_message(self):
        # input_message = ChatCompletionUserMessageParam(
        #     role="user", content=self.state.input_message
        # )
        # return input_message
        return self.state.experimental_input_file

    @listen(start_with_user_message)
    def chatbot_message(self, user_prompt):
        groq_output = self.groq(user_prompt=user_prompt)
        return groq_output

    def groq(self,
            user_prompt: ChatCompletionUserMessageParam,
            system_prompt: Optional[ChatCompletionSystemMessageParam] = None
    ) -> str:
        try:
            if not system_prompt:
                messages = [user_prompt]
            else:
                messages = [system_prompt, user_prompt]

            completion = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                temperature=0.7,
                max_completion_tokens=8192,
                top_p=1,
                stream=False,
                stop=None
            )

            print(completion.choices[0].message)
            output = completion.choices[0].message
            return output.content
        except Exception as ex:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error in groq_client: {ex}")

chat_bot = ChatbotWorkflow()


test = FastAPI()

class UserPrompt(BaseModel):
    role: str = "user"
    content: str


@test.post("/v0")
async def start_with_something(payload: UserPrompt):
    try:
        user_completions_testing = ChatCompletionUserMessageParam(
            role=payload.role, content=payload.content
        )
        inputs = {
            "experimental_input_file":user_completions_testing
        }
        content = await run_in_threadpool(chat_bot.kickoff, inputs=inputs)

        return ChatCompletionAssistantMessageParam(role="assistant",content=content)
    except Exception as ex:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error occurred: {ex}")


if __name__ == "__main__":

    import uvicorn
    print("RUNNING UVICORN")
    uvicorn.run(
        "main_flow:test",
        host="0.0.0.0",
        port=8888,
        reload=True
    )