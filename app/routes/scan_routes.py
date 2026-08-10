"""
Scan analysis route — accepts an image upload, runs the CV pipeline,
persists the result, and returns the analysis to the mobile app.
"""

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_auth
from app.database import Scan, get_db
from app.services.vitiligo_analyzer import analyse_image

router = APIRouter(prefix="/api", tags=["scan"])


@router.post("/analyzeScan")
async def analyze_scan(
    image: UploadFile = File(...),
    userId: str = Form(...),
    bodyLocation: str = Form(...),
    _auth: dict = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    image_bytes = await image.read()
    result = analyse_image(image_bytes)

    scan_id = str(uuid.uuid4())
    scan_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    scan = Scan(
        id=scan_id,
        user_id=userId,
        body_location=bodyLocation,
        scan_date=scan_date,
        affected_percent=result.affected_percent,
        unaffected_percent=result.unaffected_percent,
        patch_count=result.patch_count,
        skin_tone=result.skin_tone,
        ratio=result.ratio,
        bounding_boxes=json.dumps(result.bounding_boxes),
        detection_confidence=result.detection_confidence,
    )
    db.add(scan)
    await db.commit()

    return {
        "id": scan_id,
        "userId": userId,
        "bodyLocation": bodyLocation,
        "scanDate": scan_date,
        "affectedPercent": result.affected_percent,
        "unaffectedPercent": result.unaffected_percent,
        "patchCount": result.patch_count,
        "skinTone": result.skin_tone,
        "ratio": result.ratio,
        "boundingBoxes": result.bounding_boxes,
        "annotatedImage": result.annotated_image_b64,
        "detectionConfidence": result.detection_confidence,
        "imageUrl": "",
        "dominantColor": "#FFFFFF",
        "accentColor": "#4F46E5",
    }
