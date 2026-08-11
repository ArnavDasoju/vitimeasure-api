"""
Scan analysis route — accepts an image upload, runs the CV pipeline,
persists the result, and returns the analysis to the mobile app.
"""

import json
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_auth
from app.database import Scan, get_db
from app.services.vitiligo_analyzer import analyse_image

router = APIRouter(prefix="/api", tags=["scan"])

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/analyzeScan")
async def analyze_scan(
    image: UploadFile = File(...),
    userId: str = Form(...),
    bodyLocation: str = Form(...),
    auth: dict = Depends(require_auth),
    db: AsyncSession = Depends(get_db),
):
    # Validate userId matches authenticated user
    if auth["userId"] != userId:
        raise HTTPException(status_code=403, detail="Cannot create scans for another user")

    image_bytes = await image.read()

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="Image too large (max 20 MB)")

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="File must be an image (JPEG or PNG)")

    try:
        result = analyse_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=422, detail="Could not process image. Please try a different photo.")

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
