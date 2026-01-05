from fastapi import HTTPException, status
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from fastapi.concurrency import run_in_threadpool



ph = PasswordHasher()

async def hash_password(password: str) -> str:
    return await run_in_threadpool(ph.hash, password)

async def verify_password(hash: str, password: str) -> bool:
    try:
        return await run_in_threadpool(ph.verify, hash, password)

    except (VerificationError, VerifyMismatchError, InvalidHashError):
        return False
