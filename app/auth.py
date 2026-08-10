"""
JWT helpers and authentication dependency for FastAPI routes.
"""

from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from app.config import settings

ALGORITHM = "HS256"

bearer_scheme = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def create_token(user_id: str, email: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_expires_in_days)
    payload = {"userId": user_id, "email": email, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])


async def require_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    """FastAPI dependency — returns the decoded JWT payload or raises 401."""
    if creds is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    try:
        payload = decode_token(creds.credentials)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return payload
