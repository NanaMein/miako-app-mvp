from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class LLMSettings(BaseSettings):
    GROQ_API_KEY: SecretStr

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings_for_workflow = LLMSettings()

print(settings_for_workflow.GROQ_API_KEY)