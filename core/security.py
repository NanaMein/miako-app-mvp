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
    "domain":settings.DOMAIN
}

ph = PasswordHasher()

async def get_hash_password(password: str) -> str:
    return await run_in_threadpool(ph.hash, password)

async def verify_hash_password(hash_password: str, password: str) -> bool:
    try:
        return await run_in_threadpool(ph.verify, hash_password, password)

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
    response.set_cookie(key="access_token",value=access_token, max_age=604800, **COOKIE_SETTINGS)


def set_refresh_cookie(response: Response, subject: Union[str, Any]):
    refresh_token = create_refresh_token(subject=subject)
    response.set_cookie(key="refresh_token",value=refresh_token, max_age=604800, **COOKIE_SETTINGS)


def get_current_user_id(request: Request, response: Response):

    token_from_cookie = request.cookies.get("access_token")
    payload = token_decoder(token=token_from_cookie)

    if payload and payload.get("type") == "access":
        return payload.get("sub")

    get_refresh_token_from_cookie = request.cookies.get("refresh_token")
    refresh_token_load = token_decoder(token=get_refresh_token_from_cookie)

    if refresh_token_load and refresh_token_load.get("type") == "refresh":
        subject = refresh_token_load.get("sub")
        set_access_cookie(response=response, subject=subject)
        return subject

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated"
    )

def login_with_access_and_refresh_token(subject: Union[str, Any], response: Response):
    set_access_cookie(response=response, subject=subject)
    set_refresh_cookie(response=response, subject=subject)

def logout_and_delete_cookies(response: Response):
    response.delete_cookie(key="access_token",**COOKIE_SETTINGS)
    response.delete_cookie(key="refresh_token",**COOKIE_SETTINGS)








# SECURITY = HTTPBearer(auto_error=False)
# PASSWORD = PasswordHasher()
#
# async def get_hash_password(password: str) -> str:
#     return await run_in_threadpool(ph.hash, password)
#
# async def verify_hash_password(hash_password: str, password: str) -> bool:
#     try:
#         return await run_in_threadpool(ph.verify, hash_password, password)
#
#     except (VerificationError, VerifyMismatchError, InvalidHashError):
#         return False
#
# class PasswordHandler:
#     def __init__(self, password: str, hash_password: str | None = None):
#         self._password = password
#         self._hash_password = hash_password
#         self.ph = PASSWORD
#
#
#
#     @property
#     def password(self):
#         return self._password
#
#     @property
#     def hash_password(self):
#         return self._hash_password
#
#     async def get_hash_password(self) -> str:
#         return await run_in_threadpool(ph.hash, self.password)
#
#     async def verify_hash_password(self) -> bool:
#         try:
#             if not self.hash_password:
#                 return False
#
#             return await run_in_threadpool(ph.verify, self.hash_password, self.password)
#
#         except (VerificationError, VerifyMismatchError, InvalidHashError):
#             return False
#
# class JwtHandler:
#     def __init__(self, subject: str | dict[str, str], request: Request | None = None):
#         self._subject = subject
#         self.request = request
#         self.security = SECURITY
#
#     @property
#     def subject(self):
#         return self._subject
#
#     def _token_generator(self, token_type: str = "access"):
#         if not isinstance(self.subject, str):
#             sub = str(self.subject)
#         else:
#             sub = self.subject
#
#         time_now = datetime.now(timezone.utc)
#         if token_type == "access".strip().lower():
#             expire_time = time_now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
#             token_type = "access"
#         else:
#             expire_time = time_now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
#             token_type = "refresh"
#
#         to_encode = {
#             "sub": sub,
#             "exp": expire_time,
#             "type": token_type,
#             "iat": time_now
#         }
#         return jwt.encode(
#             to_encode, key=settings.SECRET_KEY, algorithm=settings.ALGORITHM
#         )
#
#     @staticmethod
#     def _token_decoder(token: str | None) ->  None | dict[str, Any]:
#         if not token:
#             return None
#
#         try:
#             return jwt.decode(
#                 token=token,
#                 key=settings.SECRET_KEY,
#                 algorithms=[settings.ALGORITHM]
#             )
#
#         except (JWTError, ExpiredSignatureError, JWTClaimsError):
#             return None
#
#     @staticmethod
#     def get_token_from_header(authorization: str | None = None) -> str | None:
#         if not authorization:
#             return None
#         parts = authorization.split()
#         if len(parts) != 2 or parts[0].lower() != "bearer":
#             return None
#         return parts[1]
#
#     def get_current_user_id(self, credentials: HTTPAuthorizationCredentials = Depends(SECURITY)):
#         if not credentials:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Authorization header missing",
#                 headers={"WWW-Authenticate": "Bearer"}
#             )
#
#         token = credentials.credentials
#         payload = self._token_decoder(token=token)
#
#         if not payload:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Invalid token",
#                 headers={"WWW-Authenticate": "Bearer"}
#             )
#
#         if payload.get("type") != "access":
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Token type must be access",
#                 headers={"WWW-Authenticate": "Bearer"}
#             )
#
#         return payload.get("sub")
#
#     def validate_refresh_token(self):
#         refresh_token = self.request.headers.get("X-Refresh-Token")
#         if not refresh_token:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Refresh token missing",
#                 headers={"WWW-Authenticate": "Bearer"}
#             )
#
#         payload = self._token_decoder(token=refresh_token)
#
#         if not payload or payload.get("type") != "refresh":
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="Invalid or expired refresh token",
#                 headers={"WWW-Authenticate": "Bearer"}
#             )
#
#         return payload.get("sub")
#
#     def login_response_tokens(self) -> dict[str, str]:
#         return {
#             "access_token": self._create_access_token(),
#             "refresh_token": self._create_refresh_token(),
#             "token_type": "bearer"
#         }
#
#     def refresh_access_token(self) -> str:
#         try:
#             self.validate_refresh_token()
#         except HTTPException:
#             raise
#
#         return self._create_access_token()
#
#     def _create_access_token(self) -> str:
#         jwt_token = self._token_generator(token_type="access")
#         return jwt_token
#
#     def _create_refresh_token(self) -> str:
#         jwt_token = self._token_generator(token_type="refresh")
#         return jwt_token