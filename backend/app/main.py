from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = FastAPI(title="Bio-Digital Twin - Phase 1", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data", exist_ok=True)

@app.get("/")
def root():
    return {"message": "Bio-Digital Twin Phase 1 API is running!"}

@app.get("/health")
def health():
    return {"status": "healthy", "phase": "1 - Synthetic Patient Generator"}

@app.post("/generate-patients")
def generate_patients(n: int = Query(500, ge=10, le=5000)):
    """Generate synthetic patients"""
    np.random.seed(42)
    
    data = {
        "patient_id": [f"P{str(i).zfill(6)}" for i in range(1, n+1)],
        "age": np.random.normal(45, 15, n).astype(int).clip(18, 85),
        "gender": np.random.choice(["Male", "Female"], n, p=[0.52, 0.48]),
        "bmi": np.random.normal(26.5, 5.5, n).clip(15, 45),
        "hr": np.random.normal(72, 8, n).astype(int).clip(50, 110),
        "hrv": np.random.normal(45, 18, n).astype(int).clip(10, 120),
        "spo2": np.random.normal(96.5, 1.8, n).clip(88, 100),
        "glucose": np.random.normal(105, 25, n).astype(int).clip(70, 250),
        "systolic_bp": np.random.normal(125, 15, n).astype(int).clip(90, 180),
        "diastolic_bp": np.random.normal(78, 10, n).astype(int).clip(60, 110),
        "created_at": datetime.now().isoformat()
    }
    
    df = pd.DataFrame(data)
    output_path = "data/synthetic_patients.csv"
    df.to_csv(output_path, index=False)
    
    return {
        "status": "success",
        "generated": n,
        "sample": df.head(3).to_dict(orient="records"),
        "file_saved": output_path
    }