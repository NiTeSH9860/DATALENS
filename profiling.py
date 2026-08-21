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

from numeric_cleaning import is_messy_numeric_column, clean_numeric_column


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
            # Check for messy-numeric formatting (currency, thousand
            # separators, percentages, accounting negatives) BEFORE
            # trying datetime — a column of "$1,234.56" values should
            # be classified numeric, not misread as text or dropped
            # into a low-signal categorical bucket.
            if is_messy_numeric_column(series):
                col_lower = str(col).lower()
                looks_like_id = any(kw in col_lower for kw in ["id", "code", "zip", "postal"])
                if not looks_like_id:
                    column_types[col] = "numeric"
                    continue

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


def clean_dataframe(df: pd.DataFrame, column_types: dict) -> pd.DataFrame:
    """Returns a NEW DataFrame with columns actually converted to match
    their detected type — messy-numeric text columns become real
    floats, datetime-detected columns become real datetime64. This is
    what visualize.py and qa.py should operate on, not the raw upload,
    since detect_column_types alone only classifies, it doesn't convert."""
    cleaned = df.copy()

    for col, col_type in column_types.items():
        if col_type == "numeric" and not pd.api.types.is_numeric_dtype(cleaned[col]):
            cleaned[col] = clean_numeric_column(cleaned[col])
        elif col_type == "datetime" and not pd.api.types.is_datetime64_any_dtype(cleaned[col]):
            cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce", format="mixed")

    return cleaned


def profile_dataset(df: pd.DataFrame) -> dict:
    """Build a full profile: shape, column types, missing values, and
    type-appropriate summary stats for each column."""
    column_types = detect_column_types(df)
    df = clean_dataframe(df, column_types)  # so stats below use real numbers, not strings

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
    print("=== Test: synthetic messy-numeric column (currency, thousand-separators) ===")
    messy_df = pd.DataFrame({
        "category": ["Electronics", "Furniture", "Electronics", "Furniture", "Electronics", "Furniture"],
        "revenue": ["$1,234.56", "$999.00", "$2,100.50", "$450.25", "$300.00", "$1,500.75"],
        "growth_pct": ["12%", "-5%", "8%", "20%", "3%", "15%"],
    })
    column_types = detect_column_types(messy_df)
    print(f"Detected types: {column_types}")
    assert column_types["revenue"] == "numeric", f"Expected numeric, got {column_types['revenue']}"
    assert column_types["growth_pct"] == "numeric", f"Expected numeric, got {column_types['growth_pct']}"
    assert column_types["category"] == "categorical", f"Expected categorical, got {column_types['category']}"

    cleaned = clean_dataframe(messy_df, column_types)
    print(f"Cleaned revenue values: {cleaned['revenue'].tolist()}")
    assert cleaned["revenue"].iloc[0] == 1234.56
    expected_sum = 1234.56 + 999.00 + 2100.50 + 450.25 + 300.00 + 1500.75
    assert abs(cleaned["revenue"].sum() - expected_sum) < 0.01
    print("PASSED — messy currency correctly detected AND converted to real numbers\n")

    # Quick self-test against a real CSV, purely to prove the profiling
    # logic works end-to-end on real, messy data
    import sys
    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/test.csv"
    df = pd.read_csv(test_path)
    profile = profile_dataset(df)
    print_profile_summary(profile)
