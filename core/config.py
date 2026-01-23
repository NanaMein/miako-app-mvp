from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "databases" / "test.db"
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
SQLITE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH.as_posix()}"


class Settings(BaseSettings):
    PROJECT_NAME: str = "Miako App MVP"
    SECRET_KEY: str = Field(...)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 5
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    DOMAIN: str = Field(...)
    DATABASE_TYPE: str = Field(default="sqlite") #Change this to postgres if you want to use postgres database
    _POSTGRES_URL: SecretStr

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def DATABASE_URL(self) -> str:
        if self.DATABASE_TYPE == "sqlite":
            return SQLITE_URL
        return self._POSTGRES_URL.get_secret_value()

settings = Settings()