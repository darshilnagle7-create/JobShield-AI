
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fake_job_posting_model.pkl"
HTML_PATH = BASE_DIR / "index.html"

app = FastAPI(title="Fake Job Posting Detection")

# Load trained ML model
model = joblib.load(MODEL_PATH)


class Job(BaseModel):
    title: str
    company_profile: str = ""
    description: str = ""
    requirements: str = ""

    employment_type: str = "unknown"
    required_experience: str = "unknown"
    required_education: str = "unknown"

    telecommuting: int = 0
    has_company_logo: int = 0
    has_questions: int = 0




from pathlib import Path
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent
HTML_PATH = BASE_DIR / "index.html"

@app.get("/")
def home():
    if not HTML_PATH.is_file():
        return {
            "error": "index.html not found",
            "expected_path": str(HTML_PATH)
        }

    return FileResponse(str(HTML_PATH))

@app.get("/health")
def health():
    return {"status": "running", "model_loaded": model is not None}


@app.post("/predict")
def predict(job: Job):
    text = (
        f"{job.title} {job.company_profile} "
        f"{job.description} {job.requirements}"
    )

    row = pd.DataFrame([{
        "text": text,
        "employment_type": job.employment_type,
        "required_experience": job.required_experience,
        "required_education": job.required_education,
        "telecommuting": job.telecommuting,
        "has_company_logo": job.has_company_logo,
        "has_questions": job.has_questions
    }])

    probability = float(model.predict_proba(row)[0][1])

    return {
        "fraudulent": int(probability > 0.5),
        "probability": round(probability, 2)
    }
