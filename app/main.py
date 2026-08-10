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
    return {"status": "ok", "service": "vitimeasure-api"}
