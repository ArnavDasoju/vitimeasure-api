"""
AI routes — chat and insight generation via OpenAI.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import require_auth
from app.config import settings
from app.services.openai_service import ask_ai, generate_insights

router = APIRouter(prefix="/api", tags=["ai"])


class AskAIBody(BaseModel):
    question: str
    context: dict
    conversationHistory: list[dict] = []


class ScanEntry(BaseModel):
    scanDate: str
    affectedPercent: float
    bodyLocation: str


@router.post("/askAI")
async def ask_ai_route(body: AskAIBody, _auth: dict = Depends(require_auth)):
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="AI features are not configured")
    answer = await ask_ai(body.question, body.context, body.conversationHistory)
    return {"answer": answer}


@router.post("/generateInsights")
async def generate_insights_route(scans: list[ScanEntry], _auth: dict = Depends(require_auth)):
    if not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="AI features are not configured")
    insight = await generate_insights([s.model_dump() for s in scans])
    return {"insight": insight}
