"""
Auth routes: register, login, delete account.
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import create_token, hash_password, require_auth, verify_password
from app.database import (
    CheckIn, DailyStress, Patch, Scan, Treatment, User, get_db,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterBody(BaseModel):
    email: EmailStr
    password: str
    name: str = ""


class LoginBody(BaseModel):
    email: EmailStr
    password: str


@router.post("/register")
async def register(body: RegisterBody, db: AsyncSession = Depends(get_db)):
    email = body.email.strip().lower()

    result = await db.execute(select(User).where(User.email == email))
    if result.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already in use")

    user_id = str(uuid.uuid4())
    user = User(
        id=user_id,
        email=email,
        password_hash=hash_password(body.password),
        name=body.name.strip() or email.split("@")[0],
    )
    db.add(user)
    await db.commit()

    token = create_token(user_id, email)
    return {"token": token, "userId": user_id}


@router.post("/login")
async def login(body: LoginBody, db: AsyncSession = Depends(get_db)):
    email = body.email.strip().lower()

    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_token(user.id, email)
    return {"token": token, "userId": user.id}


@router.delete("/account")
async def delete_account(
    payload: dict = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    user_id = payload["userId"]

    for model in [Scan, Patch, CheckIn, DailyStress, Treatment]:
        await db.execute(delete(model).where(model.user_id == user_id))
    await db.execute(delete(User).where(User.id == user_id))
    await db.commit()

    return {"deleted": True}
