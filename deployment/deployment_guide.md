# Deployment Guide

## Local (FastAPI)
```bash
pip install -r requirements.txt
python -m src.train --tune --output models/
uvicorn deployment.app:app --port 8000
# Visit http://localhost:8000/docs
```

## Docker
```bash
docker build -t workpulse-api .
docker run -p 8000:8000 workpulse-api
```

## GCP Vertex AI
See `deploy_vertex.sh` and `vertex_app.py` in this directory.
