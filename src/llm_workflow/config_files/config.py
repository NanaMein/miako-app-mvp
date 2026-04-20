from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class WorkflowSetting(BaseSettings):
    GROQ_API_KEY: SecretStr
    CLIENT_USER: SecretStr
    CLIENT_PASSWORD: SecretStr
    CLIENT_SERVER_NAME: SecretStr
    CLIENT_URI: SecretStr
    CLIENT_TOKEN: SecretStr
    COHERE_API_KEY: SecretStr

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

workflow_settings = WorkflowSetting()

