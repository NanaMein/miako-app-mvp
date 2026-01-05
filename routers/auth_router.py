from fastapi import APIRouter, HTTPException, status, Depends, Response, Request
from sqlmodel import select
from databases.database import get_session, AsyncSession
from models.user_model import User
from schemas.user_schema import UserBase, UserCreate, UserRead, UserLogin
from core.security import hash_password, verify_password
from pydantic import BaseModel


router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)



@router.post("/sign-up", status_code=status.HTTP_201_CREATED, response_model=UserRead)
async def sign_up_user(payload: UserCreate, session: AsyncSession = Depends(get_session)):
    hashed_password = await hash_password(payload.password)

    db_user = User(
        email=payload.email,
        user_name=payload.username,
        hashed_password=hashed_password
    )
    session.add(db_user)
    await session.commit()
    await session.refresh(db_user)
    return db_user

@router.post("/login", status_code=status.HTTP_200_OK, response_model=UserRead)
async def login_user(payload: UserLogin, session: AsyncSession = Depends(get_session)):
    if not payload.email or not payload.password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email and password is required")

    statement = select(User).where(User.email == payload.email)
    result = await session.execute(statement=statement)
    user = result.scalar_one_or_none()

    error_401 = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    if not user:
        raise error_401

    is_valid = await verify_password(user.hashed_password, payload.password)
    if not is_valid:
        raise error_401

    return user

@router.post("/log-out")
async def logout_user(response: Response):
    response.delete_cookie(
        key="access_token",
        httponly=True,
        samesite="lax",
        secure=True
    )
    return {"detail":"Successfully log out your account"}

