"""
OpenAI integration for AI chat and insight generation.
"""

from openai import AsyncOpenAI

from app.config import settings

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


SYSTEM_PROMPT = (
    "You are a knowledgeable dermatology assistant embedded in VITImeasure, "
    "a vitiligo tracking app. You help users understand their scan data, "
    "VASI scores, and trends. Always remind users that you provide "
    "informational guidance only — not medical advice — and they should "
    "consult a dermatologist for diagnosis or treatment decisions. "
    "Keep answers concise (2–4 sentences) unless asked for detail."
)


async def ask_ai(
    question: str,
    context: dict,
    conversation_history: list[dict],
) -> str:
    client = _get_client()

    context_block = (
        f"Body location: {context.get('bodyLocation', 'unknown')}. "
        f"VASI score: {context.get('vasiScore', 'N/A')}. "
        f"Change from last scan: {context.get('changeFromLast', 'N/A')}. "
        f"Total scans: {len(context.get('scans', []))}."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"[Patient context: {context_block}]"},
    ]

    for msg in conversation_history:
        role = "assistant" if msg.get("role") == "ai" else "user"
        messages.append({"role": role, "content": msg["content"]})

    messages.append({"role": "user", "content": question})

    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""


async def generate_insights(
    scans: list[dict],
) -> str:
    client = _get_client()

    scan_summary = "\n".join(
        f"- {s.get('scanDate', '?')}: {s.get('bodyLocation', '?')} — "
        f"{s.get('affectedPercent', '?')}% affected"
        for s in scans
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Based on the following scan history, provide a brief insight "
                "(2–3 sentences) about the patient's vitiligo progression trend. "
                "Be encouraging but honest.\n\n" + scan_summary
            ),
        },
    ]

    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_tokens=256,
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""
