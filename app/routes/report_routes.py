"""
Report generation route — returns scan data formatted for PDF export.

The mobile app generates the actual PDF client-side via expo-print.
This endpoint aggregates the data the client needs.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth import require_auth
from app.database import get_container

router = APIRouter(prefix="/api", tags=["report"])


class ReportBody(BaseModel):
    userId: str
    bodyLocation: str
    startDate: str
    endDate: str


@router.post("/generateReport")
async def generate_report(body: ReportBody, _auth: dict = Depends(require_auth)):
    container = get_container("scans")
    query = (
        "SELECT * FROM c WHERE c.userId = @uid AND c.bodyLocation = @loc "
        "AND c.scanDate >= @start AND c.scanDate <= @end "
        "ORDER BY c.scanDate ASC"
    )
    params = [
        {"name": "@uid", "value": body.userId},
        {"name": "@loc", "value": body.bodyLocation},
        {"name": "@start", "value": body.startDate},
        {"name": "@end", "value": body.endDate},
    ]
    scans = list(container.query_items(query, parameters=params, enable_cross_partition_query=True))

    # The mobile app expects a reportUrl but generates PDFs locally.
    # Return the data payload so the client can render its own PDF.
    return {
        "reportUrl": "",
        "scans": [
            {
                "id": s["id"],
                "scanDate": s["scanDate"],
                "affectedPercent": s.get("affectedPercent", 0),
                "unaffectedPercent": s.get("unaffectedPercent", 100),
                "patchCount": s.get("patchCount", 0),
                "skinTone": s.get("skinTone", ""),
                "boundingBoxes": s.get("boundingBoxes", []),
            }
            for s in scans
        ],
    }
