from datetime import datetime, timezone, timedelta
from typing import Optional, Any, Union
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError, JWTClaimsError
from core.config import settings
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from fastapi.concurrency import run_in_threadpool
from fastapi import Request, Response, HTTPException, status

COOKIE_SETTINGS={
    "httponly":True,
    "secure":True,
    "samesite":"lax",
    "domain":settings.MDOMAIN
}

ph = PasswordHasher()

async def get_hash_password(password: str) -> str:
    return await run_in_threadpool(ph.hash, password)

async def verify_hash_password(hash: str, password: str) -> bool:
    try:
        return await run_in_threadpool(ph.verify, hash, password)

    except (VerificationError, VerifyMismatchError, InvalidHashError):
        return False

def token_generator(sub: Union[str, Any], token_type: str = "access"):
    subject = str(sub)
    time_now = datetime.now(timezone.utc)
    if token_type == "access".strip().lower():
        expire_time = time_now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        token_type = "access"
    else:
        expire_time = time_now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        token_type = "refresh"

    to_encode = {
        "sub": subject,
        "exp": expire_time,
        "type":token_type,
        "iat": time_now
    }
    return jwt.encode(
        to_encode, key=settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )

def token_decoder(token: Optional[str]) -> Optional[dict[str, Any]]:
    if not token:
        return None

    try:
        payload: dict[str, Any] = jwt.decode(
            token=token,
            key=settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload

    except (JWTError, ExpiredSignatureError, JWTClaimsError):
        return None



def create_access_token(subject: Union[str, Any]) -> str:
    jwt_token = token_generator(sub=subject, token_type="access")
    return jwt_token

def create_refresh_token(subject: Union[str, Any]) -> str:
    jwt_token = token_generator(sub=subject, token_type="refresh")
    return jwt_token

def set_access_cookie(response: Response, subject: Union[str, Any]):
    access_token = create_access_token(subject=subject)
    response.set_cookie(key="access_token",value=access_token, **COOKIE_SETTINGS)


def set_refresh_cookie(response: Response, subject: Union[str, Any]):
    refresh_token = create_refresh_token(subject=subject)
    response.set_cookie(key="refresh_token",value=refresh_token, **COOKIE_SETTINGS)


def get_current_user_id(request: Request, response: Response):

    token_from_cookie = request.cookies.get("access_token")
    payload = token_decoder(token=token_from_cookie)
    if payload:
        return payload.get("sub")

    get_refresh_token_from_cookie = request.cookies.get("refresh_token")
    refresh_token_load = token_decoder(token=get_refresh_token_from_cookie)

    if refresh_token_load:
        subject = refresh_token_load.get("sub")
        set_access_cookie(response=response, subject=subject)
        return subject

    return None

