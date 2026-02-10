from fastapi import FastAPI
from core.config import settings
from routers.llm_workflow_router import router as llm_router


app = FastAPI()
app.include_router(llm_router)


if __name__ == "__main__":
    import uvicorn
    print("RUNNING UVICORN")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )