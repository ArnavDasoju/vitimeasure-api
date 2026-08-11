"""
VITImeasure API — FastAPI application entry point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes.auth_routes import router as auth_router
from app.routes.scan_routes import router as scan_router
from app.routes.progress_routes import router as progress_router
from app.routes.report_routes import router as report_router
from app.routes.ai_routes import router as ai_router
from app.routes.sync_routes import router as sync_router

app = FastAPI(title="VITImeasure API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(auth_router)
app.include_router(scan_router)
app.include_router(progress_router)
app.include_router(report_router)
app.include_router(ai_router)
app.include_router(sync_router)


@app.get("/")
async def health():
    from app.config import settings
    db_set = bool(settings.database_url)
    db_host = settings.database_url.split("@")[1].split("/")[0] if "@" in settings.database_url else "not set"
    return {"status": "ok", "service": "vitimeasure-api", "db_configured": db_set, "db_host": db_host}


@app.post("/api/init-db")
async def init_db():
    """One-time endpoint to create tables. Call once after deploy."""
    try:
        from app.database import create_tables
        await create_tables()
        return {"status": "tables created"}
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}
