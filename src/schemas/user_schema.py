from sqlmodel import SQLModel, Field
import uuid


class UserBase(SQLModel):
    email: str = Field(index=True, unique=True)

class UserCreate(UserBase):
    username: str
    password: str

class UserRead(UserBase):
    uuid: uuid.UUID

class UserLogin(UserBase):
    password: str