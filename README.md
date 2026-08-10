# VITImeasure API

Backend for the VITImeasure mobile app. Handles authentication, vitiligo image analysis (OpenCV), cloud sync, AI insights, and report data.

## Prerequisites

- Python 3.11+
- Azure Cosmos DB account
- OpenAI API key (optional, for AI chat/insights)

## Setup

```bash
cd vitimeasure-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in COSMOS_DB_CONNECTION_STRING, JWT_SECRET, and optionally OPENAI_API_KEY
```

## Run locally

```bash
uvicorn app.main:app --reload --port 3000
```

## Deploy to Azure App Service

```bash
# Login and create resources
az login
az group create --name vitimeasure-rg --location eastus2
az appservice plan create --name vitimeasure-plan --resource-group vitimeasure-rg --sku B1 --is-linux
az webapp create --name vitimeasure-api --resource-group vitimeasure-rg --plan vitimeasure-plan --runtime "PYTHON:3.11"

# Set environment variables
az webapp config appsettings set --name vitimeasure-api --resource-group vitimeasure-rg --settings \
  COSMOS_DB_CONNECTION_STRING="<your-connection-string>" \
  COSMOS_DB_DATABASE_NAME="vitiligo_db" \
  JWT_SECRET="<random-64-char-string>" \
  OPENAI_API_KEY="<your-key>"

# Set startup command
az webapp config set --name vitimeasure-api --resource-group vitimeasure-rg --startup-file "startup.sh"

# Deploy
az webapp up --name vitimeasure-api --resource-group vitimeasure-rg --runtime "PYTHON:3.11"
```

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
| POST | `/api/sync/checkins` | Sync weekly check-ins |
| POST | `/api/sync/daily-stress` | Sync daily stress entries |
| POST | `/api/sync/treatments` | Sync treatment logs |
