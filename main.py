from typing import Optional

from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from routers.message_router import router as message_router
from routers.auth_router import router as auth_router

load_dotenv()

app = FastAPI()
app.include_router(message_router)
app.include_router(auth_router)

class HelloWorld(BaseModel):
    message: Optional[str] = None
    is_it_success: bool = True

@app.post("/", response_model=HelloWorld, status_code=status.HTTP_200_OK)
def hello_world(hello: HelloWorld):
    if not hello.message:
        hello.message = os.getenv("HELLO")

    if not hello.is_it_success:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No Hello World")
    else:
        return hello



if __name__ == "__main__":
    import uvicorn
    print("RUNNING UVICORN")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )