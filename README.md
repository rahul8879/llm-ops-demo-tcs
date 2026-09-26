# Production RAG Policy Assistant

LangChain + OpenAI + FastAPI RAG demo for manual deployment to GCP Cloud Run.

## Local
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
set -a; source .env; set +a
uvicorn app.main:app --reload --port 8080
```

Open `http://localhost:8080/docs`.

## Docker
```bash
docker build -t production-rag:v1 .
docker run --rm -p 8080:8080 --env-file .env production-rag:v1
```
