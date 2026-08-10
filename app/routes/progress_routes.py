"""
Progress route — returns scan history for a user + body location.
"""

from fastapi import APIRouter, Depends, Query

from app.auth import require_auth
from app.database import get_container

router = APIRouter(prefix="/api", tags=["progress"])


@router.get("/getProgress")
async def get_progress(
    userId: str = Query(...),
    bodyLocation: str = Query(...),
    _auth: dict = Depends(require_auth),
):
    container = get_container("scans")
    query = (
        "SELECT * FROM c WHERE c.userId = @uid AND c.bodyLocation = @loc "
        "ORDER BY c.scanDate ASC"
    )
    params = [
        {"name": "@uid", "value": userId},
        {"name": "@loc", "value": bodyLocation},
    ]
    items = list(container.query_items(query, parameters=params, enable_cross_partition_query=True))

    return [
        {
            "id": item["id"],
            "userId": item["userId"],
            "bodyLocation": item["bodyLocation"],
            "scanDate": item["scanDate"],
            "affectedPercent": item.get("affectedPercent", 0),
            "unaffectedPercent": item.get("unaffectedPercent", 100),
            "imageUrl": item.get("imageUrl", ""),
            "boundingBoxes": item.get("boundingBoxes", []),
            "dominantColor": item.get("dominantColor", "#FFFFFF"),
            "accentColor": item.get("accentColor", "#4F46E5"),
            "vasiScore": item.get("vasiScore"),
        }
        for item in items
    ]
