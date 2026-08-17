"""
DataLens — Natural Language Q&A
--------------------------------------
Gemini interprets a natural-language question about the uploaded data
and outputs a STRUCTURED operation spec (JSON) — not executable code.
That spec is then run through a fixed, safe set of pandas operations.

Why not just have the LLM write pandas code and eval() it: the data is
user-uploaded and arbitrary, and the question is user-typed and
arbitrary. Executing LLM-generated code against that combination is a
real code-injection risk. Constraining the LLM to a small, validated
set of operations (filter, groupby+agg, sort, describe, count, corr)
means the "attack surface" is a handful of well-tested functions, not
an open Python interpreter.

Setup:
    pip install google-genai python-dotenv pandas
    .env: GEMINI_API_KEY=your_key_here
"""

import json
import os

import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"

ALLOWED_OPERATIONS = {"filter_groupby_agg", "sort_top_n", "describe_column", "row_count", "correlation"}
ALLOWED_AGGS = {"sum", "mean", "count", "min", "max", "median"}


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            pass
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in .env or Streamlit secrets.")
    return genai.Client(api_key=api_key)


PROMPT_TEMPLATE = """You are a data analyst assistant. Given a dataset's schema and a
user's question, output ONLY a JSON object (no markdown fences, no extra text)
describing which of these fixed operations answers the question:

{{
  "operation": one of "filter_groupby_agg" | "sort_top_n" | "describe_column" | "row_count" | "correlation",
  "group_by_column": <column name to group by, or null>,
  "value_column": <numeric column to aggregate/describe, or null>,
  "agg": one of "sum" | "mean" | "count" | "min" | "max" | "median", or null,
  "filter_column": <column name to filter on, or null>,
  "filter_value": <value to filter for, or null>,
  "sort_ascending": true or false,
  "top_n": <integer, default 10>,
  "explanation": <one sentence, in plain English, of what this computes>
}}

Rules:
- Only reference column names that actually exist in the schema below.
- If the question can't be answered with these operations, set "operation" to null
  and explain why in "explanation".
- group_by_column, value_column, filter_column, agg can all be null if not needed
  for the chosen operation.

DATASET SCHEMA:
{schema}

USER QUESTION:
{question}
"""


def _build_schema_summary(df: pd.DataFrame, column_types: dict) -> str:
    lines = []
    for col, col_type in column_types.items():
        lines.append(f"- {col} ({col_type})")
    return "\n".join(lines)


def interpret_question(df: pd.DataFrame, column_types: dict, question: str) -> dict:
    """Send the question + schema to Gemini, get back a structured
    operation spec. Validates the operation name against the allowlist
    before returning — an unrecognized operation is treated as a
    failure, never passed through to execution."""
    client = _get_client()
    schema = _build_schema_summary(df, column_types)
    prompt = PROMPT_TEMPLATE.format(schema=schema, question=question)

    response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
    raw_text = response.text.strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[len("json"):].strip()

    spec = json.loads(raw_text)

    if spec.get("operation") is not None and spec["operation"] not in ALLOWED_OPERATIONS:
        raise ValueError(f"Model returned an operation outside the allowed set: {spec.get('operation')}")

    # Validate every referenced column actually exists — never trust
    # the model's column names blindly, even though we asked it to
    # only use real ones
    for key in ["group_by_column", "value_column", "filter_column"]:
        col = spec.get(key)
        if col is not None and col not in df.columns:
            raise ValueError(f"Model referenced a column that doesn't exist: {col}")

    if spec.get("agg") is not None and spec["agg"] not in ALLOWED_AGGS:
        raise ValueError(f"Model returned an aggregation outside the allowed set: {spec.get('agg')}")

    return spec


def execute_operation(df: pd.DataFrame, spec: dict) -> dict:
    """Runs the validated spec through a fixed set of safe pandas
    operations. Every branch here is a well-tested, bounded function —
    no eval, no exec, no arbitrary code path."""
    op = spec.get("operation")

    if op is None:
        return {"type": "explanation_only", "explanation": spec.get("explanation", "")}

    if op == "row_count":
        return {"type": "scalar", "value": len(df), "explanation": spec.get("explanation")}

    if op == "filter_groupby_agg":
        working = df
        if spec.get("filter_column") and spec.get("filter_value") is not None:
            working = working[working[spec["filter_column"]] == spec["filter_value"]]

        if spec.get("group_by_column") and spec.get("value_column") and spec.get("agg"):
            result = working.groupby(spec["group_by_column"])[spec["value_column"]].agg(spec["agg"])
            result_df = result.reset_index()
            result_df.columns = [spec["group_by_column"], f"{spec['agg']}_{spec['value_column']}"]
            return {"type": "table", "data": result_df, "explanation": spec.get("explanation")}

        if spec.get("value_column") and spec.get("agg"):
            value = getattr(working[spec["value_column"]], spec["agg"])()
            return {"type": "scalar", "value": round(float(value), 2), "explanation": spec.get("explanation")}

        return {"type": "table", "data": working, "explanation": spec.get("explanation")}

    if op == "sort_top_n":
        col = spec.get("value_column")
        n = spec.get("top_n", 10)
        ascending = spec.get("sort_ascending", False)
        if col is None:
            raise ValueError("sort_top_n requires a value_column")
        result_df = df.sort_values(col, ascending=ascending).head(n)
        return {"type": "table", "data": result_df, "explanation": spec.get("explanation")}

    if op == "describe_column":
        col = spec.get("value_column") or spec.get("group_by_column")
        if col is None:
            raise ValueError("describe_column requires a column")
        return {"type": "table", "data": df[col].describe().reset_index(), "explanation": spec.get("explanation")}

    if op == "correlation":
        numeric_df = df.select_dtypes(include="number")
        return {"type": "table", "data": numeric_df.corr().reset_index(), "explanation": spec.get("explanation")}

    raise ValueError(f"Unhandled operation: {op}")


def answer_question(df: pd.DataFrame, column_types: dict, question: str) -> dict:
    spec = interpret_question(df, column_types, question)
    result = execute_operation(df, spec)
    result["spec"] = spec
    return result
