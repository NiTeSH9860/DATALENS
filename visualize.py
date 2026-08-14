"""
DataLens — Auto-Visualization
------------------------------------
Given a DataFrame and its profile (from profiling.py), automatically
generates a sensible set of charts based on what's actually in the
data — no LLM involved, purely rule-based on detected column types.

Run: python visualize.py data/your_file.csv (saves charts to preview/)
Requires: pip install pandas plotly
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from profiling import detect_column_types, profile_dataset

MAX_CATEGORICAL_BARS = 15  # cap bar chart categories so it stays readable
MAX_NUMERIC_HISTOGRAMS = 6  # cap how many numeric distributions we auto-plot


def build_visualizations(df: pd.DataFrame, column_types: dict) -> list[dict]:
    """Returns a list of {"title": str, "figure": plotly Figure} dicts —
    the Streamlit app just iterates this and calls st.plotly_chart."""
    charts = []

    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in column_types.items() if t == "categorical"]
    datetime_cols = [c for c, t in column_types.items() if t == "datetime"]

    # Numeric distributions
    for col in numeric_cols[:MAX_NUMERIC_HISTOGRAMS]:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        charts.append({"title": f"Distribution of {col}", "figure": fig})

    # Categorical counts (skip columns with too many unique values —
    # would be an unreadable bar chart, not a useful one)
    for col in categorical_cols:
        n_unique = df[col].nunique()
        if n_unique > MAX_CATEGORICAL_BARS:
            continue
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", title=f"Count by {col}")
        charts.append({"title": f"Count by {col}", "figure": fig})

    # Correlation heatmap, if there are enough numeric columns for it
    # to mean anything
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=["#d84c4c", "#f5f5f5", "#2fa84f"], zmid=0,
            text=corr.round(2).values, texttemplate="%{text}",
        ))
        fig.update_layout(title="Correlation Between Numeric Columns")
        charts.append({"title": "Correlation Between Numeric Columns", "figure": fig})

    # Time trend, if there's a datetime column and at least one numeric
    # column to trend against
    if datetime_cols and numeric_cols:
        date_col = datetime_cols[0]
        value_col = numeric_cols[0]
        trend_df = df[[date_col, value_col]].copy()
        trend_df[date_col] = pd.to_datetime(trend_df[date_col], errors="coerce", format="mixed")
        trend_df = trend_df.dropna(subset=[date_col]).sort_values(date_col)
        if len(trend_df) > 0:
            fig = px.line(trend_df, x=date_col, y=value_col, title=f"{value_col} over {date_col}")
            charts.append({"title": f"{value_col} over {date_col}", "figure": fig})

    return charts


if __name__ == "__main__":
    import sys
    import os

    test_path = sys.argv[1] if len(sys.argv) > 1 else "data/superstore.csv"
    df = pd.read_csv(test_path)
    column_types = detect_column_types(df)

    charts = build_visualizations(df, column_types)
    print(f"Generated {len(charts)} charts from {len(df.columns)} columns:")
    for c in charts:
        print(f"  - {c['title']}")

    os.makedirs("preview", exist_ok=True)
    for i, c in enumerate(charts):
        path = f"preview/chart_{i}_{c['title'][:30].replace(' ', '_').replace('/', '-')}.png"
        c["figure"].write_image(path, width=800, height=500)
    print(f"\nSaved {len(charts)} chart images to preview/ for review")
