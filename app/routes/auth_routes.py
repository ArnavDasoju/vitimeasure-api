"""
Auth routes: register, login, delete account.
"""

import uuid

from azure.cosmos.exceptions import CosmosResourceNotFoundError
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from app.auth import create_token, hash_password, require_auth, verify_password
from app.database import get_container

router = APIRouter(prefix="/api/auth", tags=["auth"])


class RegisterBody(BaseModel):
    email: EmailStr
    password: str
    name: str = ""


class LoginBody(BaseModel):
    email: EmailStr
    password: str


# ─── Register ────────────────────────────────────────────────────────────────

@router.post("/register")
async def register(body: RegisterBody):
    container = get_container("users")
    email = body.email.strip().lower()

    # Check if email already exists
    query = "SELECT * FROM c WHERE c.email = @email"
    params = [{"name": "@email", "value": email}]
    existing = list(container.query_items(query, parameters=params, enable_cross_partition_query=True))
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already in use")

    user_id = str(uuid.uuid4())
    user_doc = {
        "id": user_id,
        "email": email,
        "passwordHash": hash_password(body.password),
        "name": body.name.strip() or email.split("@")[0],
        "createdAt": _now_iso(),
    }
    container.create_item(user_doc)

    token = create_token(user_id, email)
    return {"token": token, "userId": user_id}


# ─── Login ───────────────────────────────────────────────────────────────────

@router.post("/login")
async def login(body: LoginBody):
    container = get_container("users")
    email = body.email.strip().lower()

    query = "SELECT * FROM c WHERE c.email = @email"
    params = [{"name": "@email", "value": email}]
    results = list(container.query_items(query, parameters=params, enable_cross_partition_query=True))

    if not results:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    user = results[0]
    if not verify_password(body.password, user["passwordHash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_token(user["id"], email)
    return {"token": token, "userId": user["id"]}


# ─── Delete account ──────────────────────────────────────────────────────────

@router.delete("/account")
async def delete_account(payload: dict = Depends(require_auth)):
    user_id = payload["userId"]
    email = payload["email"]

    # Delete user document
    users = get_container("users")
    try:
        users.delete_item(item=user_id, partition_key=email)
    except CosmosResourceNotFoundError:
        pass

    # Delete all user data from every container
    for container_name in ["scans", "patches", "check_ins", "daily_stress", "treatments"]:
        container = get_container(container_name)
        query = "SELECT c.id FROM c WHERE c.userId = @uid"
        params = [{"name": "@uid", "value": user_id}]
        items = list(container.query_items(query, parameters=params, enable_cross_partition_query=True))
        for item in items:
            try:
                container.delete_item(item=item["id"], partition_key=user_id)
            except CosmosResourceNotFoundError:
                pass

    return {"deleted": True}


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
