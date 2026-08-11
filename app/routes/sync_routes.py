"""
Sync routes — bidirectional sync for patches, scans, check-ins, stress, and treatments.

Response shapes match what cloudSync.ts expects:
  GET /sync/patches        → { patches: [...] }
  GET /sync/scans/:patchId → { scans: [...] }
  GET /sync/checkins       → { checkIns: [...] }
  GET /sync/daily-stress   → { entries: [...] }
  GET /sync/treatments/:patchId → { treatments: [...] }
"""

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import require_auth
from app.database import (
    CheckIn, DailyStress, Patch, Scan, Treatment, get_db,
)

router = APIRouter(prefix="/api/sync", tags=["sync"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── Patches ─────────────────────────────────────────────────────────────────

class PatchItemIn(BaseModel):
    id: str
    userId: str
    bodyLocation: str
    createdAt: str | None = None
    scans: list[dict] = []


class PatchSyncBody(BaseModel):
    patches: list[PatchItemIn]


@router.post("/patches")
async def sync_patches(body: PatchSyncBody, auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    for p in body.patches:
        existing = await db.get(Patch, p.id)
        if existing:
            existing.body_location = p.bodyLocation
            existing.synced_at = _now_iso()
        else:
            db.add(Patch(
                id=p.id, user_id=p.userId, body_location=p.bodyLocation,
                created_at=p.createdAt or _now_iso(), synced_at=_now_iso(),
            ))
    await db.commit()
    return {"synced": len(body.patches)}


@router.get("/patches")
async def get_patches(auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    user_id = auth["userId"]
    result = await db.execute(select(Patch).where(Patch.user_id == user_id))
    patches = result.scalars().all()
    return {"patches": [
        {"id": p.id, "userId": p.user_id, "bodyLocation": p.body_location,
         "createdAt": p.created_at, "syncedAt": p.synced_at}
        for p in patches
    ]}


@router.delete("/patches/{patch_id}")
async def delete_patch(patch_id: str, auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    await db.execute(delete(Patch).where(Patch.id == patch_id, Patch.user_id == auth["userId"]))
    await db.commit()
    return {"deleted": True}


# ─── Scans ───────────────────────────────────────────────────────────────────

class ScanItemIn(BaseModel):
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
    scans: list[ScanItemIn]


@router.post("/scans")
async def sync_scans(body: ScanSyncBody, auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    for s in body.scans:
        existing = await db.get(Scan, s.id)
        if existing:
            existing.synced_at = _now_iso()
        else:
            db.add(Scan(
                id=s.id, user_id=s.userId, patch_id=s.patchId,
                body_location="", scan_date=s.date,
                affected_percent=s.affectedPercent, unaffected_percent=s.unaffectedPercent,
                vasi_score=s.vasiScore, image_url=s.imageUrl or "",
                bounding_boxes=json.dumps(s.boundingBoxes),
                dominant_color=s.dominantColor or "#FFFFFF",
                accent_color=s.accentColor or "#4F46E5",
                synced_at=_now_iso(),
            ))
    await db.commit()
    return {"synced": len(body.scans)}


@router.get("/scans/{patch_id}")
async def get_scans_for_patch(patch_id: str, auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    user_id = auth["userId"]
    result = await db.execute(
        select(Scan).where(Scan.user_id == user_id, Scan.patch_id == patch_id).order_by(Scan.scan_date.asc())
    )
    scans = result.scalars().all()
    return {"scans": [
        {"id": s.id, "patchId": s.patch_id, "userId": s.user_id, "date": s.scan_date,
         "affectedPercent": s.affected_percent, "unaffectedPercent": s.unaffected_percent,
         "vasiScore": s.vasi_score, "imageUrl": s.image_url,
         "boundingBoxes": json.loads(s.bounding_boxes) if s.bounding_boxes else [],
         "dominantColor": s.dominant_color, "accentColor": s.accent_color}
        for s in scans
    ]}


# ─── Check-ins ───────────────────────────────────────────────────────────────

class CheckInItemIn(BaseModel):
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
    checkIns: list[CheckInItemIn]


@router.post("/checkins")
async def sync_checkins(body: CheckInSyncBody, auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    for ci in body.checkIns:
        existing = await db.get(CheckIn, ci.id)
        if existing:
            existing.stress_score = ci.stressScore
            existing.sleep_score = ci.sleepScore
            existing.mood_score = ci.moodScore
            existing.synced_at = _now_iso()
        else:
            db.add(CheckIn(
                id=ci.id, user_id=ci.userId, week_key=ci.weekKey,
                week_start_date=ci.weekStartDate, stress_score=ci.stressScore,
                sleep_score=ci.sleepScore, mood_score=ci.moodScore,
                completed_at=ci.completedAt, synced_at=_now_iso(),
            ))
    await db.commit()
    return {"synced": len(body.checkIns)}


@router.get("/checkins")
async def get_checkins(auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    user_id = auth["userId"]
    result = await db.execute(
        select(CheckIn).where(CheckIn.user_id == user_id).order_by(CheckIn.week_start_date.asc())
    )
    items = result.scalars().all()
    return {"checkIns": [
        {"id": c.id, "userId": c.user_id, "weekKey": c.week_key,
         "weekStartDate": c.week_start_date, "stressScore": c.stress_score,
         "sleepScore": c.sleep_score, "moodScore": c.mood_score,
         "completedAt": c.completed_at, "syncedAt": c.synced_at}
        for c in items
    ]}


# ─── Daily stress ────────────────────────────────────────────────────────────

class DailyStressItemIn(BaseModel):
    id: str
    userId: str
    date: str
    stressScore: int
    source: str = "in-app"
    loggedAt: str


class DailyStressSyncBody(BaseModel):
    entries: list[DailyStressItemIn]


@router.post("/daily-stress")
async def sync_daily_stress(body: DailyStressSyncBody, auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    for e in body.entries:
        existing = await db.get(DailyStress, e.id)
        if existing:
            existing.stress_score = e.stressScore
            existing.synced_at = _now_iso()
        else:
            db.add(DailyStress(
                id=e.id, user_id=e.userId, date=e.date,
                stress_score=e.stressScore, source=e.source,
                logged_at=e.loggedAt, synced_at=_now_iso(),
            ))
    await db.commit()
    return {"synced": len(body.entries)}


@router.get("/daily-stress")
async def get_daily_stress(auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    user_id = auth["userId"]
    result = await db.execute(
        select(DailyStress).where(DailyStress.user_id == user_id).order_by(DailyStress.date.asc())
    )
    items = result.scalars().all()
    return {"entries": [
        {"id": e.id, "userId": e.user_id, "date": e.date,
         "stressScore": e.stress_score, "source": e.source,
         "loggedAt": e.logged_at, "syncedAt": e.synced_at}
        for e in items
    ]}


# ─── Treatments ──────────────────────────────────────────────────────────────

class TreatmentItemIn(BaseModel):
    id: str
    userId: str
    patchId: str = ""
    bodyLocation: str = ""
    # Mobile app sends medicationName/dose; accept both naming conventions
    name: str = ""
    medicationName: str = ""
    dosage: str = ""
    dose: str = ""
    startDate: str = ""
    endDate: str | None = None
    notes: str = ""
    createdAt: str = ""
    updatedAt: str = ""
    isActive: bool = True

    @property
    def resolved_name(self) -> str:
        return self.medicationName or self.name

    @property
    def resolved_dosage(self) -> str:
        return self.dose or self.dosage


class TreatmentSyncBody(BaseModel):
    treatments: list[TreatmentItemIn]


@router.post("/treatments")
async def sync_treatments(body: TreatmentSyncBody, auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    for t in body.treatments:
        existing = await db.get(Treatment, t.id)
        if existing:
            existing.name = t.resolved_name
            existing.dosage = t.resolved_dosage
            existing.synced_at = _now_iso()
        else:
            db.add(Treatment(
                id=t.id, user_id=t.userId, patch_id=t.patchId,
                body_location=t.bodyLocation, name=t.resolved_name, dosage=t.resolved_dosage,
                start_date=t.startDate, end_date=t.endDate,
                notes=t.notes, synced_at=_now_iso(),
            ))
    await db.commit()
    return {"synced": len(body.treatments)}


@router.get("/treatments/{patch_id}")
async def get_treatments_for_patch(patch_id: str, auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    user_id = auth["userId"]
    result = await db.execute(
        select(Treatment).where(Treatment.user_id == user_id, Treatment.patch_id == patch_id)
    )
    items = result.scalars().all()
    return {"treatments": [
        {"id": t.id, "userId": t.user_id, "patchId": t.patch_id,
         "bodyLocation": t.body_location, "name": t.name, "dosage": t.dosage,
         "startDate": t.start_date, "endDate": t.end_date, "notes": t.notes}
        for t in items
    ]}


@router.delete("/treatments/{treatment_id}")
async def delete_treatment(treatment_id: str, auth: dict = Depends(require_auth), db: AsyncSession = Depends(get_db)):
    await db.execute(delete(Treatment).where(Treatment.id == treatment_id, Treatment.user_id == auth["userId"]))
    await db.commit()
    return {"deleted": True}
