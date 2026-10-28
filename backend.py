
import json
import os
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse

import joblib
import numpy as np
from scipy.sparse import hstack
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "phishing_url_model_v3.joblib"
VECTORIZER_PATH = ROOT / "models" / "phishing_url_vectorizer_v3.joblib"
INVESTIGATION_PATH = ROOT / "latest_investigation.json"

API_TOKEN = os.getenv("EXTENSION_API_TOKEN", "browser-test-token-123")

app = FastAPI(title="Browser Forensics AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def require_token(authorization: str | None):
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Invalid API token")


def extract_url_features(url):
    url = str(url).strip()
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    return [
        len(url), len(domain), len(path), len(query),
        domain.count("."), domain.count("-"), domain.count("_"),
        url.count("/"), url.count("?"), url.count("="),
        url.count("&"), url.count("%"),
        sum(c.isdigit() for c in url),
        sum(c.isalpha() for c in url),
        int(url.startswith("https://")),
        int(url.startswith("http://")),
        int("@" in url), int("://" in url), len(set(domain)),
    ]


class HistoryRecord(BaseModel):
    url: HttpUrl
    title: str = ""
    visitTime: str = ""
    visitCount: int = 0


class AnalysisRequest(BaseModel):
    records: list[HistoryRecord]


class InvestigationRequest(BaseModel):
    start_date: str
    end_date: str
    record_count: int
    analyzed_url_count: int
    phishing_count: int
    benign_count: int
    status: str = "completed"
    analysis: list[dict]


@app.get("/")
def home():
    return {"status": "online", "service": "Browser Forensics AI"}


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="V3 model files are missing from the models folder."
        )

    if not request.records:
        return {"total": 0, "results": []}

    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)

        records = [record.model_dump() for record in request.records]
        urls = [str(record["url"]) for record in records]

        text_features = vectorizer.transform(urls)
        numeric_features = np.asarray(
            [extract_url_features(url) for url in urls]
        )
        features = hstack([text_features, numeric_features]).tocsr()

        predictions = model.predict(features)
        probabilities = model.predict_proba(features)
        results = []

        for record, prediction, probs in zip(
            records, predictions, probabilities
        ):
            class_index = list(model.classes_).index(prediction)
            results.append({
                **record,
                "predicted_class": str(prediction),
                "status": "PHISHING" if prediction == 1 else "BENIGN",
                "confidence_percent": round(
                    float(probs[class_index]) * 100, 2
                ),
                "label_mapping_verified": False
            })

        return {"total": len(results), "results": results}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@app.post("/api/history/save")
def save_investigation(
    request: InvestigationRequest,
    authorization: str | None = Header(default=None),
):
    require_token(authorization)

    investigation = request.model_dump()
    investigation["investigation_id"] = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d%H%M%S")
    investigation["saved_at"] = datetime.now(timezone.utc).isoformat()

    try:
        INVESTIGATION_PATH.write_text(
            json.dumps(investigation, indent=2),
            encoding="utf-8"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not save investigation: {exc}"
        )

    return {"saved": True, "investigation_id": investigation["investigation_id"]}


@app.get("/api/history/latest")
def latest_investigation(
    authorization: str | None = Header(default=None),
):
    require_token(authorization)

    if not INVESTIGATION_PATH.exists():
        raise HTTPException(status_code=404, detail="No investigation saved yet.")

    try:
        return json.loads(INVESTIGATION_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read saved investigation: {exc}"
        )
