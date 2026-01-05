import os
from  pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Miako App MVP"
    SECRET_KEY: str= os.getenv("SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 5
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    class Config:
        env_file = ".env"

settings = Settings()