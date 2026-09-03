"""HTTP API for the React DataLens frontend."""

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from profiling import clean_dataframe, detect_column_types, profile_dataset
from QA import answer_question
from robust_loader import load_csv_robustly

app = FastAPI(title="DataLens API")
allowed_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


def _analyze(raw_bytes: bytes) -> tuple[dict, dict, object]:
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    try:
        dataframe, warnings = load_csv_robustly(raw_bytes)
        column_types = detect_column_types(dataframe)
        dataframe = clean_dataframe(dataframe, column_types)
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {error}") from error

    profile = profile_dataset(dataframe)
    preview = dataframe.head(20).where(dataframe.notna(), None).to_dict(orient="records")
    return {
        "profile": profile,
        "column_types": column_types,
        "preview": preview,
        "warnings": [warning.message for warning in warnings],
    }, column_types, dataframe


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    result, _, _ = _analyze(await file.read())
    result["filename"] = file.filename
    return result


@app.post("/api/ask")
async def ask(file: UploadFile = File(...), question: str = Form(...)) -> dict:
    if not question.strip():
        raise HTTPException(status_code=400, detail="A question is required.")
    _, column_types, dataframe = _analyze(await file.read())
    try:
        result = answer_question(dataframe, column_types, question.strip())
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    if result.get("type") == "table":
        result["data"] = result["data"].where(result["data"].notna(), None).to_dict(orient="records")
    return result
