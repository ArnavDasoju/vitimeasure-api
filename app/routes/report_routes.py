"""
Report generation route — returns scan data formatted for the client PDF export.
"""

import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_auth
from app.database import Scan, get_db

router = APIRouter(prefix="/api", tags=["report"])


class ReportBody(BaseModel):
    userId: str
    bodyLocation: str
    startDate: str
    endDate: str


@router.post("/generateReport")
async def generate_report(
    body: ReportBody,
    _auth: dict = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Scan)
        .where(
            Scan.user_id == body.userId,
            Scan.body_location == body.bodyLocation,
            Scan.scan_date >= body.startDate,
            Scan.scan_date <= body.endDate,
        )
        .order_by(Scan.scan_date.asc())
    )
    scans = result.scalars().all()

    return {
        "reportUrl": "",
        "scans": [
            {
                "id": s.id,
                "scanDate": s.scan_date,
                "affectedPercent": s.affected_percent,
                "unaffectedPercent": s.unaffected_percent,
                "patchCount": s.patch_count,
                "skinTone": s.skin_tone,
                "boundingBoxes": json.loads(s.bounding_boxes) if s.bounding_boxes else [],
            }
            for s in scans
        ],
    }
