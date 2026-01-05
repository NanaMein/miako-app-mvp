from fastapi import FastAPI
from routers.auth_router import router as auth_router

app = FastAPI()
app.include_router(auth_router)


if __name__ == "__main__":
    import uvicorn
    print("RUNNING UVICORN")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )