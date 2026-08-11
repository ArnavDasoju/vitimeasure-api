# VITImeasure API

Backend for the VITImeasure mobile app. Handles authentication, vitiligo image analysis (OpenCV), cloud sync, AI insights, and report data.

## Prerequisites

- Python 3.11+
- Supabase account (free tier PostgreSQL)
- OpenAI API key (optional, for AI chat/insights)

## Setup

```bash
cd vitimeasure-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in DATABASE_URL, JWT_SECRET, and optionally OPENAI_API_KEY
```

## Run locally

```bash
uvicorn app.main:app --reload --port 3000
```

Then call `POST /api/init-db` once to create tables.

## Deploy to Render

The repo includes `render.yaml` for auto-configuration. Push to GitHub, connect to Render, and set environment variables:

- `DATABASE_URL` — Supabase session pooler connection string
- `JWT_SECRET` — random 64-character string
- `OPENAI_API_KEY` — optional

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/auth/register` | Create account |
| POST | `/api/auth/login` | Sign in (returns JWT) |
| DELETE | `/api/auth/account` | Delete account + all data |
| POST | `/api/analyzeScan` | Upload image, run CV analysis |
| GET | `/api/getProgress` | Scan history for a body location |
| POST | `/api/generateReport` | Aggregate scan data for PDF |
| POST | `/api/askAI` | AI chat about scan data |
| POST | `/api/generateInsights` | AI trend insights |
| POST | `/api/sync/patches` | Sync patches |
| GET | `/api/sync/patches` | Get user's patches |
| POST | `/api/sync/scans` | Sync scans |
| GET | `/api/sync/scans/:patchId` | Get scans for a patch |
| POST | `/api/sync/checkins` | Sync weekly check-ins |
| POST | `/api/sync/daily-stress` | Sync daily stress entries |
| POST | `/api/sync/treatments` | Sync treatment logs |
