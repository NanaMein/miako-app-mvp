from datetime import datetime, timezone, timedelta
from typing import Any
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError, JWTClaimsError
from core.config import settings
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from fastapi.concurrency import run_in_threadpool
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import HTTPException, status, Depends



security = HTTPBearer(auto_error=False)
ph = PasswordHasher()

async def get_hash_password(password: str) -> str:
    return await run_in_threadpool(ph.hash, password)

async def verify_hash_password(hash_password: str, password: str) -> bool:
    try:
        return await run_in_threadpool(ph.verify, hash_password, password)

    except (VerificationError, VerifyMismatchError, InvalidHashError):
        return False

def token_generator(sub: str| Any, token_type: str = "access"):
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
        to_encode, key=settings.SECRET_KEY.get_secret_value(), algorithm=settings.ALGORITHM
    )

def token_decoder(token: str | None) -> dict[str, Any] | None:
    if not token:
        return None

    try:
        return jwt.decode(
            token=token,
            key=settings.SECRET_KEY.get_secret_value(),
            algorithms=[settings.ALGORITHM]
        )

    except (JWTError, ExpiredSignatureError, JWTClaimsError):
        return None



def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)):
        if not credentials:
            raise TokenException(
                detail="Authorization header missing",
                headers={"WWW-Authenticate": "Bearer"}
            )

        token = credentials.credentials
        payload = token_decoder(token=token)

        if not payload:
            raise TokenException(
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"}
            )

        if payload.get("type") != "access":
            raise TokenException(
                detail="Token type must be access",
                headers={"WWW-Authenticate": "Bearer"}
            )

        return payload.get("sub")


def get_access_token_by_refresh_token(refresh_token: str):
    try:
        payload = token_decoder(token=refresh_token)

        if not payload.get("type") == "refresh":
            raise TokenException(detail="Invalid refresh token type")

        exp = payload.get("exp")
        if exp is None or datetime.now(timezone.utc).timestamp() > exp:
            raise TokenException(detail="Refresh token has expired")

        user_id = payload.get("sub")  # or whatever claim you use
        if not user_id:
            raise TokenException(detail="Invalid token payload")

        new_access_token = create_access_token(subject=user_id)
        return new_access_token

    except (JWTError, ExpiredSignatureError, JWTClaimsError):
        raise TokenException(detail="Invalid refresh token")
    except Exception as ex:
        raise TokenException(detail=f"Token refresh failed: {str(ex)}")




def login_response_tokens(subject: str | Any) -> dict[str, str]:
    return {
        "access_token": create_access_token(subject=subject),
        "refresh_token": create_refresh_token(subject=subject),
        "token_type": "bearer"
    }

def create_access_token(subject: str | Any) -> str:
    jwt_token = token_generator(sub=subject, token_type="access")
    return jwt_token

def create_refresh_token(subject: str | Any) -> str:
    jwt_token = token_generator(sub=subject, token_type="refresh")
    return jwt_token

class TokenException(HTTPException):
    def __init__(self, detail: str = "Unauthorized", **kwargs: Any):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            **kwargs
        )