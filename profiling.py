"""
CSV Insights Tool — Data Profiling
------------------------------------------
Given any uploaded CSV, automatically figures out what's actually in
it: column types (numeric / categorical / datetime / text), missing
values, and basic summary stats — the foundation the auto-visualization
and LLM Q&A layers will both build on.

No LLM involved here — this is pure pandas, deterministic, and testable
independent of any API key or network access.

Run: python profiling.py (tests against a sample CSV)
Requires: pip install pandas
"""

import pandas as pd


def detect_column_types(df: pd.DataFrame) -> dict:
    """Classify each column as numeric, categorical, datetime, or text.
    Datetime detection tries to parse object columns as dates before
    falling back to categorical/text, since pandas doesn't auto-detect
    dates stored as plain strings in a CSV."""
    column_types = {}

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            # Numeric dtype doesn't always mean a meaningful continuous
            # variable — postal codes, zip codes, and ID columns are
            # numeric but semantically identifiers, not something to
            # histogram or correlate. Flag likely identifiers by name
            # pattern or by "every value is unique" cardinality.
            col_lower = str(col).lower()
            looks_like_id = any(kw in col_lower for kw in ["id", "code", "zip", "postal"])
            n_unique = series.nunique()
            all_unique = len(series) > 0 and n_unique == len(series)

            if looks_like_id or all_unique:
                column_types[col] = "identifier"
            else:
                column_types[col] = "numeric"
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            column_types[col] = "datetime"
            continue

        # Newer pandas versions (2.x+ with string inference, or 3.x)
        # default to a dedicated StringDtype for string columns instead
        # of the legacy generic 'object' dtype — checking dtype == "object"
        # alone silently misses these. is_object_dtype/is_string_dtype
        # together catch both cases.
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Try parsing as a date — if most non-null values parse
            # successfully, treat it as a datetime column
            sample = series.dropna().head(50)
            if len(sample) > 0:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                success_rate = parsed.notna().mean()
                if success_rate > 0.8:
                    column_types[col] = "datetime"
                    continue

            # Categorical vs. free text: low cardinality relative to row
            # count suggests categories, high cardinality suggests text
            n_unique = series.nunique()
            n_total = len(series)
            if n_total > 0 and (n_unique / n_total) < 0.5 and n_unique < 100:
                column_types[col] = "categorical"
            else:
                column_types[col] = "text"
            continue

        column_types[col] = "other"

    return column_types


def profile_dataset(df: pd.DataFrame) -> dict:
    """Build a full profile: shape, column types, missing values, and
    type-appropriate summary stats for each column."""
    column_types = detect_column_types(df)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)

    column_profiles = {}
    for col, col_type in column_types.items():
        profile = {
            "type": col_type,
            "missing_count": int(missing[col]),
            "missing_pct": float(missing_pct[col]),
        }

        if col_type == "numeric":
            profile.update({
                "mean": round(float(df[col].mean()), 2) if df[col].notna().any() else None,
                "median": round(float(df[col].median()), 2) if df[col].notna().any() else None,
                "std": round(float(df[col].std()), 2) if df[col].notna().any() else None,
                "min": round(float(df[col].min()), 2) if df[col].notna().any() else None,
                "max": round(float(df[col].max()), 2) if df[col].notna().any() else None,
            })
        elif col_type == "categorical":
            value_counts = df[col].value_counts().head(5)
            profile["top_values"] = value_counts.to_dict()
            profile["n_unique"] = int(df[col].nunique())
        elif col_type == "datetime":
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            profile["min_date"] = str(parsed.min()) if parsed.notna().any() else None
            profile["max_date"] = str(parsed.max()) if parsed.notna().any() else None

        column_profiles[col] = profile

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "columns": column_profiles,
    }


def print_profile_summary(profile: dict) -> None:
    print(f"Dataset: {profile['n_rows']} rows, {profile['n_columns']} columns\n")

    for col, info in profile["columns"].items():
        print(f"  {col} ({info['type']})")
        if info["missing_count"] > 0:
            print(f"    Missing: {info['missing_count']} ({info['missing_pct']}%)")
        if info["type"] == "numeric":
            print(f"    Range: {info['min']} to {info['max']}, mean: {info['mean']}")
        elif info["type"] == "categorical":
            top = list(info["top_values"].items())[:3]
            print(f"    {info['n_unique']} unique values. Top: {top}")
        elif info["type"] == "datetime":
            print(f"    Range: {info['min_date']} to {info['max_date']}")
        print()


if __name__ == "__main__":
    # Quick self-test against the sales dashboard's dataset, reused here
    # purely to prove the profiling logic works on a real, messy CSV
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.csv"
    df = pd.read_csv(test_path)
    profile = profile_dataset(df)
    print_profile_summary(profile)
