# Bio-Digital Twin Platform

## Phase 1: Synthetic Patient Generator

### Setup
1. `docker compose up -d`
2. `cd backend`
3. `python -m venv venv`
4. `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
5. `pip install -r requirements.txt`
6. `uvicorn app.main:app --reload`

API docs: http://localhost:8000/docs
