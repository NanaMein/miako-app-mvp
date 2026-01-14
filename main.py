from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.auth_router import router as auth_router
from core.config import settings
# from dotenv import load_dotenv
# import os
# load_dotenv()

app = FastAPI()

# origins=[os.getenv("ALLOW_ORIGIN")]
ORIGINS=settings.ALLOW_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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