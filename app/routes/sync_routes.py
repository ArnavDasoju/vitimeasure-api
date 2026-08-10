"""
Sync routes — bidirectional sync for patches, scans, check-ins, stress, and treatments.

The mobile app stores data locally and syncs to the cloud in the background.
These endpoints accept arrays of items to upsert and return the server's
current state for the user.

Response shapes must match what cloudSync.ts expects:
  GET /sync/patches        → { patches: [...] }
  GET /sync/scans/:patchId → { scans: [...] }
  GET /sync/checkins       → { checkIns: [...] }
  GET /sync/daily-stress   → { entries: [...] }
  GET /sync/treatments/:patchId → { treatments: [...] }
"""

from datetime import datetime, timezone

from azure.cosmos.exceptions import CosmosResourceNotFoundError
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.auth import require_auth
from app.database import get_container

router = APIRouter(prefix="/api/sync", tags=["sync"])


# ─── Patches ─────────────────────────────────────────────────────────────────

class PatchItem(BaseModel):
    id: str
    userId: str
    bodyLocation: str
    createdAt: str | None = None
    scans: list[dict] = []


class PatchSyncBody(BaseModel):
    patches: list[PatchItem]


@router.post("/patches")
async def sync_patches(body: PatchSyncBody, auth: dict = Depends(require_auth)):
    container = get_container("patches")
    for patch in body.patches:
        doc = patch.model_dump()
        doc["syncedAt"] = _now_iso()
        container.upsert_item(doc)
    return {"synced": len(body.patches)}


@router.get("/patches")
async def get_patches(auth: dict = Depends(require_auth)):
    user_id = auth["userId"]
    container = get_container("patches")
    query = "SELECT * FROM c WHERE c.userId = @uid"
    params = [{"name": "@uid", "value": user_id}]
    items = list(container.query_items(query, parameters=params, enable_cross_partition_query=True))
    return {"patches": items}


@router.delete("/patches/{patch_id}")
async def delete_patch(patch_id: str, auth: dict = Depends(require_auth)):
    user_id = auth["userId"]
    container = get_container("patches")
    try:
        container.delete_item(item=patch_id, partition_key=user_id)
    except CosmosResourceNotFoundError:
        pass
    return {"deleted": True}


# ─── Scans ───────────────────────────────────────────────────────────────────

class ScanItem(BaseModel):
    id: str
    patchId: str
    userId: str
    date: str
    affectedPercent: float
    unaffectedPercent: float
    vasiScore: float | None = None
    imageUrl: str | None = None
    boundingBoxes: list[dict] = []
    dominantColor: str | None = None
    accentColor: str | None = None


class ScanSyncBody(BaseModel):
    scans: list[ScanItem]


@router.post("/scans")
async def sync_scans(body: ScanSyncBody, auth: dict = Depends(require_auth)):
    container = get_container("scans")
    for scan in body.scans:
        doc = scan.model_dump()
        doc["syncedAt"] = _now_iso()
        container.upsert_item(doc)
    return {"synced": len(body.scans)}


@router.get("/scans/{patch_id}")
async def get_scans_for_patch(patch_id: str, auth: dict = Depends(require_auth)):
    user_id = auth["userId"]
    container = get_container("scans")
    query = "SELECT * FROM c WHERE c.userId = @uid AND c.patchId = @pid ORDER BY c.date ASC"
    params = [
        {"name": "@uid", "value": user_id},
        {"name": "@pid", "value": patch_id},
    ]
    items = list(container.query_items(query, parameters=params, enable_cross_partition_query=True))
    return {"scans": items}


# ─── Check-ins ───────────────────────────────────────────────────────────────

class CheckInItem(BaseModel):
    id: str
    userId: str
    weekKey: str
    weekStartDate: str
    stressScore: int
    sleepScore: int
    moodScore: int
    completedAt: str
    syncedAt: str | None = None


class CheckInSyncBody(BaseModel):
    checkIns: list[CheckInItem]


@router.post("/checkins")
async def sync_checkins(body: CheckInSyncBody, auth: dict = Depends(require_auth)):
    container = get_container("check_ins")
    for ci in body.checkIns:
        doc = ci.model_dump()
        doc["syncedAt"] = _now_iso()
        container.upsert_item(doc)
    return {"synced": len(body.checkIns)}


@router.get("/checkins")
async def get_checkins(auth: dict = Depends(require_auth)):
    user_id = auth["userId"]
    container = get_container("check_ins")
    query = "SELECT * FROM c WHERE c.userId = @uid ORDER BY c.weekStartDate ASC"
    params = [{"name": "@uid", "value": user_id}]
    items = list(container.query_items(query, parameters=params, enable_cross_partition_query=True))
    return {"checkIns": items}


# ─── Daily stress ────────────────────────────────────────────────────────────

class DailyStressItem(BaseModel):
    id: str
    userId: str
    date: str
    stressScore: int
    source: str = "in-app"
    loggedAt: str


class DailyStressSyncBody(BaseModel):
    entries: list[DailyStressItem]


@router.post("/daily-stress")
async def sync_daily_stress(body: DailyStressSyncBody, auth: dict = Depends(require_auth)):
    container = get_container("daily_stress")
    for entry in body.entries:
        doc = entry.model_dump()
        doc["syncedAt"] = _now_iso()
        container.upsert_item(doc)
    return {"synced": len(body.entries)}


@router.get("/daily-stress")
async def get_daily_stress(auth: dict = Depends(require_auth)):
    user_id = auth["userId"]
    container = get_container("daily_stress")
    query = "SELECT * FROM c WHERE c.userId = @uid ORDER BY c.date ASC"
    params = [{"name": "@uid", "value": user_id}]
    items = list(container.query_items(query, parameters=params, enable_cross_partition_query=True))
    return {"entries": items}


# ─── Treatments ──────────────────────────────────────────────────────────────

class TreatmentItem(BaseModel):
    id: str
    userId: str
    patchId: str = ""
    bodyLocation: str = ""
    name: str = ""
    dosage: str = ""
    startDate: str = ""
    endDate: str | None = None
    notes: str = ""


class TreatmentSyncBody(BaseModel):
    treatments: list[TreatmentItem]


@router.post("/treatments")
async def sync_treatments(body: TreatmentSyncBody, auth: dict = Depends(require_auth)):
    container = get_container("treatments")
    for t in body.treatments:
        doc = t.model_dump()
        doc["syncedAt"] = _now_iso()
        container.upsert_item(doc)
    return {"synced": len(body.treatments)}


@router.get("/treatments/{patch_id}")
async def get_treatments_for_patch(patch_id: str, auth: dict = Depends(require_auth)):
    user_id = auth["userId"]
    container = get_container("treatments")
    query = "SELECT * FROM c WHERE c.userId = @uid AND c.patchId = @pid"
    params = [
        {"name": "@uid", "value": user_id},
        {"name": "@pid", "value": patch_id},
    ]
    items = list(container.query_items(query, parameters=params, enable_cross_partition_query=True))
    return {"treatments": items}


@router.delete("/treatments/{treatment_id}")
async def delete_treatment(treatment_id: str, auth: dict = Depends(require_auth)):
    user_id = auth["userId"]
    container = get_container("treatments")
    try:
        container.delete_item(item=treatment_id, partition_key=user_id)
    except CosmosResourceNotFoundError:
        pass
    return {"deleted": True}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
