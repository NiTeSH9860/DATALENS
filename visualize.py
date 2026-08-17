"""
DataLens — Auto-Visualization
------------------------------------
Given a DataFrame and its profile (from profiling.py), automatically
generates a broad, general-purpose set of charts based on what's
actually in the data — no LLM involved, purely rule-based on detected
column types. Designed to work reasonably on ANY uploaded CSV, not
just datasets shaped like the ones used to build it.

Run: python visualize.py data/your_file.csv
Requires: pip install pandas plotly numpy
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from profiling import detect_column_types

MAX_CATEGORICAL_BARS = 12          # categories shown individually before bucketing into "Other"
MAX_NUMERIC_HISTOGRAMS = 8         # cap how many numeric distributions we auto-plot
MAX_BOX_PLOTS = 8                  # cap how many outlier box plots we auto-plot
MAX_SCATTER_PAIRS = 4              # cap how many numeric-pair scatter plots we auto-plot
MIN_CORRELATION_FOR_SCATTER = 0.3  # only scatter-plot pairs with at least this much |correlation|


def _bucket_top_n_plus_other(series: pd.Series, n: int) -> pd.Series:
    """For high-cardinality categoricals, keep the top N categories by
    frequency and collapse everything else into 'Other' — this keeps
    the chart readable instead of either an unreadable 50-bar chart or
    silently dropping the column entirely."""
    counts = series.value_counts()
    top_categories = set(counts.head(n).index)
    return series.apply(lambda v: v if v in top_categories else "Other")


def detect_outliers_iqr(series: pd.Series) -> dict:
    """Standard IQR-based outlier detection: anything more than 1.5x
    the interquartile range beyond Q1/Q3 is flagged. Returns counts and
    bounds so the UI can show a plain-language summary, not just a chart."""
    clean = series.dropna()
    if len(clean) < 4:
        return {"n_outliers": 0, "lower_bound": None, "upper_bound": None}

    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    n_outliers = int(((clean < lower) | (clean > upper)).sum())

    return {
        "n_outliers": n_outliers,
        "pct_outliers": round(n_outliers / len(clean) * 100, 1),
        "lower_bound": round(float(lower), 2),
        "upper_bound": round(float(upper), 2),
    }


def _top_correlated_pairs(df: pd.DataFrame, numeric_cols: list, max_pairs: int) -> list:
    """Returns the N numeric column PAIRS with the strongest absolute
    correlation, above a minimum threshold — used to pick which scatter
    plots are actually worth showing, instead of an unreadable N-choose-2
    grid for wide datasets."""
    if len(numeric_cols) < 2:
        return []

    corr = df[numeric_cols].corr(numeric_only=True).abs()
    pairs = []
    seen = set()
    for col_a in numeric_cols:
        for col_b in numeric_cols:
            if col_a == col_b:
                continue
            key = tuple(sorted([col_a, col_b]))
            if key in seen:
                continue
            seen.add(key)
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and value >= MIN_CORRELATION_FOR_SCATTER:
                pairs.append((key[0], key[1], value))

    pairs.sort(key=lambda p: -p[2])
    return pairs[:max_pairs]


def build_visualizations(df: pd.DataFrame, column_types: dict) -> list[dict]:
    """Returns a list of {"title", "category", "figure"} dicts. The
    "category" field (distribution / relationship / trend / categorical)
    lets a UI group charts into tabs instead of one long scroll."""
    charts = []

    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in column_types.items() if t == "categorical"]
    datetime_cols = [c for c, t in column_types.items() if t == "datetime"]

    # --- Distributions: histograms ---
    for col in numeric_cols[:MAX_NUMERIC_HISTOGRAMS]:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        charts.append({"title": f"Distribution of {col}", "category": "distribution", "figure": fig})

    # --- Distributions: box plots for outlier visibility ---
    for col in numeric_cols[:MAX_BOX_PLOTS]:
        fig = px.box(df, y=col, title=f"Outliers in {col}", points="outliers")
        charts.append({"title": f"Outliers in {col}", "category": "distribution", "figure": fig})

    # --- Categorical counts, with top-N + Other bucketing ---
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique > MAX_CATEGORICAL_BARS:
            bucketed = _bucket_top_n_plus_other(df[col], MAX_CATEGORICAL_BARS)
            counts = bucketed.value_counts().reset_index()
            title = f"Count by {col} (top {MAX_CATEGORICAL_BARS} + Other)"
        else:
            counts = df[col].value_counts().reset_index()
            title = f"Count by {col}"
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", title=title)
        charts.append({"title": title, "category": "categorical", "figure": fig})

    # --- Relationships: correlation heatmap ---
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=["#d84c4c", "#f5f5f5", "#2fa84f"], zmid=0,
            text=corr.round(2).values, texttemplate="%{text}",
        ))
        fig.update_layout(title="Correlation Between Numeric Columns")
        charts.append({"title": "Correlation Between Numeric Columns", "category": "relationship", "figure": fig})

    # --- Relationships: scatter plots for the most correlated pairs ---
    top_pairs = _top_correlated_pairs(df, numeric_cols, MAX_SCATTER_PAIRS)
    for col_a, col_b, corr_value in top_pairs:
        fig = px.scatter(
            df, x=col_a, y=col_b,
            title=f"{col_a} vs. {col_b} (correlation: {corr_value:.2f})",
            trendline="ols",
        )
        charts.append({
            "title": f"{col_a} vs. {col_b}", "category": "relationship", "figure": fig,
        })

    # --- Trends: numeric columns over time ---
    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        trend_df = df.copy()
        trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors="coerce", format="mixed")
        trend_df = trend_df.dropna(subset=[date_col]).sort_values(date_col)

        # Overlay up to 3 numeric columns (by variance, as a proxy for
        # "columns with an interesting shape over time") on one chart,
        # rather than one chart per numeric column which doesn't scale
        if len(trend_df) > 0:
            variances = df[numeric_cols].var(numeric_only=True).sort_values(ascending=False)
            top_value_cols = variances.head(3).index.tolist()
            fig = px.line(
                trend_df, x=date_col, y=top_value_cols,
                title=f"Trend over {date_col}",
                labels={"value": "Value", "variable": ""},
            )
            charts.append({"title": f"Trend over {date_col}", "category": "trend", "figure": fig})

    return charts


def build_outlier_summary(df: pd.DataFrame, column_types: dict) -> dict:
    """Plain-data outlier summary per numeric column, for display
    alongside (not instead of) the box plots."""
    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    return {col: detect_outliers_iqr(df[col]) for col in numeric_cols}


if __name__ == "__main__":
    import sys

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/superstore.csv"
    df = pd.read_csv(test_path)
    column_types = detect_column_types(df)

    charts = build_visualizations(df, column_types)
    print(f"Generated {len(charts)} charts from {len(df.columns)} columns:\n")

    by_category = {}
    for c in charts:
        by_category.setdefault(c["category"], []).append(c["title"])
    for category, titles in by_category.items():
        print(f"[{category}]")
        for t in titles:
            print(f"  - {t}")
        print()

    print("Outlier summary:")
    outliers = build_outlier_summary(df, column_types)
    for col, info in outliers.items():
        if info["n_outliers"] > 0:
            print(f"  {col}: {info['n_outliers']} outliers ({info['pct_outliers']}%), "
                  f"expected range [{info['lower_bound']}, {info['upper_bound']}]")
