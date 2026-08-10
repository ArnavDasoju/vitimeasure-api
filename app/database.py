"""
PostgreSQL database — async SQLAlchemy engine + session factory.

Tables are created automatically on startup via the lifespan handler.
"""

import ssl as _ssl

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text,
    func,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

_ssl_ctx: _ssl.SSLContext | None = None
if "supabase" in settings.database_url:
    _ssl_ctx = _ssl.create_default_context()
    _ssl_ctx.check_hostname = False
    _ssl_ctx.verify_mode = _ssl.CERT_NONE

_connect_args = {"ssl": _ssl_ctx} if _ssl_ctx else {}

# Engine and session are created once — they are safe to reuse across requests.
# The engine lazily creates connections, so this does not connect at import time.
engine = create_async_engine(
    settings.async_database_url,
    echo=False,
    pool_size=5,
    max_overflow=10,
    connect_args=_connect_args,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


# ─── Models ──────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    name = Column(String, default="")
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Scan(Base):
    __tablename__ = "scans"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    patch_id = Column(String, index=True)
    body_location = Column(String, nullable=False)
    scan_date = Column(String, nullable=False)
    affected_percent = Column(Float, default=0)
    unaffected_percent = Column(Float, default=100)
    patch_count = Column(Integer, default=0)
    skin_tone = Column(String, default="")
    ratio = Column(String, default="")
    bounding_boxes = Column(Text, default="[]")
    detection_confidence = Column(Float, default=0)
    image_url = Column(String, default="")
    dominant_color = Column(String, default="#FFFFFF")
    accent_color = Column(String, default="#4F46E5")
    vasi_score = Column(Float, nullable=True)
    synced_at = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Patch(Base):
    __tablename__ = "patches"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    body_location = Column(String, nullable=False)
    created_at = Column(String, default="")
    synced_at = Column(String, nullable=True)


class CheckIn(Base):
    __tablename__ = "check_ins"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    week_key = Column(String, nullable=False)
    week_start_date = Column(String, nullable=False)
    stress_score = Column(Integer, default=3)
    sleep_score = Column(Integer, default=3)
    mood_score = Column(Integer, default=3)
    completed_at = Column(String, default="")
    synced_at = Column(String, nullable=True)


class DailyStress(Base):
    __tablename__ = "daily_stress"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    date = Column(String, nullable=False)
    stress_score = Column(Integer, default=3)
    source = Column(String, default="in-app")
    logged_at = Column(String, default="")
    synced_at = Column(String, nullable=True)


class Treatment(Base):
    __tablename__ = "treatments"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    patch_id = Column(String, default="")
    body_location = Column(String, default="")
    name = Column(String, default="")
    dosage = Column(String, default="")
    start_date = Column(String, default="")
    end_date = Column(String, nullable=True)
    notes = Column(String, default="")
    synced_at = Column(String, nullable=True)


# ─── Session dependency ──────────────────────────────────────────────────────

async def get_db():
    async with async_session() as session:
        yield session


async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
