from datetime import datetime, timezone, timedelta
from typing import Optional, Any, Union
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError, JWTClaimsError
from core.config import settings
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from fastapi.concurrency import run_in_threadpool



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

def token_verification(token: str) -> Optional[dict[str, Any]]:
    try:
        payload: dict[str, Any] = jwt.decode(
            token=token,
            key=settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        user_id = payload.get("sub")
        if user_id is None:
            return None
        return payload
    except (JWTError, ExpiredSignatureError, JWTClaimsError):
        return None



def create_access_token(subject: Union[str, Any]) -> str:
    jwt_token = token_generator(sub=subject, token_type="access")
    return jwt_token

def create_refresh_token(subject: Union[str, Any]) -> str:
    jwt_token = token_generator(sub=subject, token_type="refresh")
    return jwt_token

