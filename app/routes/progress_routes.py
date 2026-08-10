"""
Progress route — returns scan history for a user + body location.
"""

import json

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_auth
from app.database import Scan, get_db

router = APIRouter(prefix="/api", tags=["progress"])


@router.get("/getProgress")
async def get_progress(
    userId: str = Query(...),
    bodyLocation: str = Query(...),
    _auth: dict = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Scan)
        .where(Scan.user_id == userId, Scan.body_location == bodyLocation)
        .order_by(Scan.scan_date.asc())
    )
    scans = result.scalars().all()

    return [
        {
            "id": s.id,
            "userId": s.user_id,
            "bodyLocation": s.body_location,
            "scanDate": s.scan_date,
            "affectedPercent": s.affected_percent,
            "unaffectedPercent": s.unaffected_percent,
            "imageUrl": s.image_url,
            "boundingBoxes": json.loads(s.bounding_boxes) if s.bounding_boxes else [],
            "dominantColor": s.dominant_color,
            "accentColor": s.accent_color,
            "vasiScore": s.vasi_score,
        }
        for s in scans
    ]
