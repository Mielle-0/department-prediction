import os
import joblib
import numpy as np
import pandas as pd
import secrets
import logging
from dotenv import load_dotenv
from pydantic import BaseModel
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import APIKeyHeader
from contextlib import asynccontextmanager

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "svc_tfidf_pipeline.joblib"
MASK_PATH = BASE_DIR / "models" / "branch_to_dep_map.joblib"

assets = {"pipeline": None, "mask": None}

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
EXPECTED_API_KEY = os.environ.get("API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Check if files exist before loading
        if not MODEL_PATH.exists() or not MASK_PATH.exists():
            raise FileNotFoundError(f"Model files not found at {MODEL_PATH.parent}")

        # Load Pipeline and Mask
        assets["pipeline"] = joblib.load(MODEL_PATH)
        assets["mask"] = joblib.load(MASK_PATH)
        
        print(f"✅ Local Assets Loaded: {MODEL_PATH.name} and {MASK_PATH.name}")
    except Exception as e:
        print(f"❌ Startup Error: {e}")
    yield
    assets.clear()
app = FastAPI(title="Modern Multi-Model API", lifespan=lifespan)


@app.get("/")
async def health_check():
    return {
        "status": "online",
        "pipeline_loaded": assets["pipeline"] is not None,
        "mask_loaded": assets["mask"] is not None
    }
    

async def verify_api_key(api_key_header: str = Security(api_key_header)):
    # Use constant-time comparison in production (e.g., secrets.compare_digest)
    if not secrets.compare_digest(api_key_header, EXPECTED_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key_header



class PredictRequest(BaseModel):
    branch: str
    details: str


class PredictionEntry(BaseModel):
    department: str
    probability: float

class PredictResponse(BaseModel):
    used_model: str
    top_3: List[PredictionEntry]


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, api_key: str = Depends(verify_api_key)):

    pipeline = assets.get("pipeline")
    branch_map = assets.get("mask")

    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not loaded")
    if not branch_map:
        raise HTTPException(status_code=404, detail="Branch mask not loaded")

    clean_branch = request.branch.strip()
    clean_details = request.details.strip()

    if not clean_details:
        raise HTTPException(status_code=400, detail="Feedback details cannot be empty")

    input_df = pd.DataFrame([{
        'branch': clean_branch, 
        'details_cleaned': clean_details
    }])

    try:
        all_probs = pipeline.predict_proba(input_df.fillna(''))[0]
        classes = pipeline.classes_

        allowed_deps = branch_map.get(request.branch, [])

        masked_candidates = []
        for idx, dep_id in enumerate(classes):
            if dep_id in allowed_deps:
                masked_candidates.append({
                    "department": str(dep_id),
                    "probability": float(all_probs[idx])
                })

        masked_candidates.sort(key=lambda x: x['probability'], reverse=True)

        results = masked_candidates[:3] # Selects only 3

        return PredictResponse(used_model=str(MODEL_PATH), top_3=results)
        
    except Exception as e:

        logging.error(f"Internal prediction error: {e}", exc_info=True)
        # Return a generic, safe response to the user
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred during prediction."
        )
