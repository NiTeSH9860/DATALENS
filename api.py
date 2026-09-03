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
    visualizations = []
    categorical_chart_count = 0
    numeric_columns = [column for column, kind in column_types.items() if kind == "numeric"]
    datetime_columns = [column for column, kind in column_types.items() if kind == "datetime"]
    for column, column_type in column_types.items():
        if column_type == "categorical":
            counts = dataframe[column].value_counts().head(8)
            visualizations.append({
                "kind": "donut" if categorical_chart_count == 0 else "bar",
                "title": f"Count by {column}",
                "column": str(column),
                "labels": [str(label) for label in counts.index],
                "values": [int(value) for value in counts.values],
            })
            categorical_chart_count += 1
        elif column_type == "numeric":
            values = dataframe[column].dropna()
            if len(values) > 0:
                bins = min(8, max(3, int(values.nunique())))
                histogram = values.value_counts(bins=bins, sort=False).sort_index()
                visualizations.append({
                    "kind": "histogram",
                    "title": f"Distribution of {column}",
                    "column": str(column),
                    "labels": [f"{interval.left:.1f}–{interval.right:.1f}" for interval in histogram.index],
                    "values": [int(value) for value in histogram.values],
                })
    if datetime_columns and numeric_columns:
        date_column, value_column = datetime_columns[0], numeric_columns[0]
        trend = dataframe[[date_column, value_column]].dropna().copy()
        trend[date_column] = trend[date_column].dt.to_period("M").astype(str)
        trend = trend.groupby(date_column)[value_column].mean().tail(12)
        visualizations.append({
            "kind": "line",
            "title": f"Average {value_column} over time",
            "column": str(value_column),
            "labels": trend.index.tolist(),
            "values": [round(float(value), 2) for value in trend.values],
        })
    if len(numeric_columns) >= 2:
        relationship = dataframe[numeric_columns[:2]].dropna().head(80)
        visualizations.append({
            "kind": "scatter",
            "title": f"{numeric_columns[0]} vs {numeric_columns[1]}",
            "column": str(numeric_columns[1]),
            "labels": [str(round(float(value), 2)) for value in relationship[numeric_columns[0]]],
            "values": [round(float(value), 2) for value in relationship[numeric_columns[1]]],
        })
    selected_visualizations = []
    selected_kinds = set()
    for visualization in visualizations:
        if visualization["kind"] not in selected_kinds:
            selected_visualizations.append(visualization)
            selected_kinds.add(visualization["kind"])
    for visualization in visualizations:
        if visualization not in selected_visualizations and len(selected_visualizations) < 8:
            selected_visualizations.append(visualization)

    preview = dataframe.head(20).where(dataframe.notna(), None).to_dict(orient="records")
    return {
        "profile": profile,
        "column_types": column_types,
        "preview": preview,
        "visualizations": selected_visualizations,
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
