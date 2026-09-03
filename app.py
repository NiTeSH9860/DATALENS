"""
DataLens — Main App
--------------------------
Ties together auth, robust file loading, profiling, auto-visualization,
Q&A, and per-user dataset storage into one Streamlit app.

Login is required before any upload/analysis/Q&A is accessible.

Run:
    streamlit run app.py
Requires: everything in requirements.txt, plus a configured .env
(DATABASE_URL, GEMINI_API_KEY)
"""

import pandas as pd
import streamlit as st

import auth
import Datasets
from profiling import detect_column_types, clean_dataframe, profile_dataset
from robust_loader import load_csv_robustly
from visualize import build_visualizations, build_outlier_summary
from QA import answer_question

st.set_page_config(page_title="DataLens", page_icon="🔎", layout="wide")

# Initialize both tables on first run — safe to call repeatedly,
# CREATE TABLE IF NOT EXISTS is idempotent
auth.init_db()
Datasets.init_db()


# ---------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------
if "user" not in st.session_state:
    st.session_state.user = None
if "current_df" not in st.session_state:
    st.session_state.current_df = None
if "current_column_types" not in st.session_state:
    st.session_state.current_column_types = None
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None
if "current_dataset_id" not in st.session_state:
    st.session_state.current_dataset_id = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []


# ---------------------------------------------------------------
# Auth gate — nothing below this renders until logged in
# ---------------------------------------------------------------
def render_login_page():
    st.title("🔎 DataLens")
    st.caption("Upload any CSV. Get instant visualizations and ask questions in plain English.")

    tab_login, tab_register = st.tabs(["Log In", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", type="primary", use_container_width=True)

        if submitted:
            try:
                user = auth.verify_user(username, password)
                st.session_state.user = user
                st.rerun()
            except auth.AuthError as e:
                st.error(str(e))

    with tab_register:
        with st.form("register_form"):
            new_username = st.text_input("Choose a username")
            new_email = st.text_input("Email")
            new_password = st.text_input("Choose a password", type="password")
            st.caption("At least 8 characters.")
            submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

        if submitted:
            try:
                auth.create_user(new_username, new_email, new_password)
                st.success("Account created! Log in from the other tab.")
            except auth.AuthError as e:
                st.error(str(e))


if st.session_state.user is None:
    render_login_page()
    st.stop()


# ---------------------------------------------------------------
# Logged-in app
# ---------------------------------------------------------------
with st.sidebar:
    st.write(f"Logged in as **{st.session_state.user['username']}**")
    if st.button("Log Out"):
        st.session_state.user = None
        st.session_state.current_df = None
        st.session_state.current_dataset_id = None
        st.rerun()

    st.divider()
    st.subheader("Upload a CSV")
    uploaded_file = st.file_uploader("Choose a file", type=["csv"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        df, load_warnings = load_csv_robustly(file_bytes)

        for w in load_warnings:
            st.warning(w.message)

        column_types = detect_column_types(df)
        df = clean_dataframe(df, column_types)

        st.session_state.current_df = df
        st.session_state.current_column_types = column_types
        st.session_state.current_filename = uploaded_file.name
        st.session_state.current_dataset_id = None  # not yet saved
        st.session_state.qa_history = []

        st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

        if st.button("💾 Save this dataset", use_container_width=True):
            profile = profile_dataset(df)
            dataset_id = Datasets.save_dataset(
                st.session_state.user["id"], uploaded_file.name, df, profile
            )
            st.session_state.current_dataset_id = dataset_id
            st.success("Saved.")
            st.rerun()

    st.divider()
    st.subheader("My Datasets")
    my_datasets = Datasets.list_datasets(st.session_state.user["id"])

    if not my_datasets:
        st.caption("No saved datasets yet.")
    else:
        for d in my_datasets:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(f"📄 {d['filename']}", key=f"load_{d['id']}", use_container_width=True):
                    loaded_df, loaded_profile = Datasets.load_dataset(d["id"], st.session_state.user["id"])
                    st.session_state.current_df = loaded_df
                    st.session_state.current_column_types = detect_column_types(loaded_df)
                    st.session_state.current_filename = d["filename"]
                    st.session_state.current_dataset_id = d["id"]
                    st.session_state.qa_history = []
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{d['id']}"):
                    Datasets.delete_dataset(d["id"], st.session_state.user["id"])
                    st.rerun()


# ---------------------------------------------------------------
# Main content
# ---------------------------------------------------------------
st.title("🔎 DataLens")

if st.session_state.current_df is None:
    st.info("Upload a CSV from the sidebar to get started, or load a saved dataset.")
    st.stop()

df = st.session_state.current_df
column_types = st.session_state.current_column_types
st.caption(f"Currently viewing: **{st.session_state.current_filename}**")

tab_overview, tab_viz, tab_qa = st.tabs(["📊 Overview", "📈 Visualizations", "💬 Ask Your Data"])

with tab_overview:
    profile = profile_dataset(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", profile["n_rows"])
    col2.metric("Columns", profile["n_columns"])
    col3.metric("Missing values", sum(c["missing_count"] for c in profile["columns"].values()))

    st.subheader("Columns")
    summary_rows = []
    for col, info in profile["columns"].items():
        summary_rows.append({
            "Column": col, "Type": info["type"],
            "Missing": f"{info['missing_count']} ({info['missing_pct']}%)",
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    outliers = build_outlier_summary(df, column_types)
    flagged = {k: v for k, v in outliers.items() if v["n_outliers"] > 0}
    if flagged:
        st.subheader("Outliers detected")
        for col, info in flagged.items():
            st.caption(f"**{col}**: {info['n_outliers']} outliers ({info['pct_outliers']}%), "
                       f"expected range [{info['lower_bound']}, {info['upper_bound']}]")

with tab_viz:
    charts = build_visualizations(df, column_types)
    if not charts:
        st.info("Not enough structured data in this file to auto-generate charts.")
    else:
        categories = sorted(set(c["category"] for c in charts))
        viz_tabs = st.tabs([c.title() for c in categories])
        for viz_tab, category in zip(viz_tabs, categories):
            with viz_tab:
                for chart in [c for c in charts if c["category"] == category]:
                    st.plotly_chart(chart["figure"], use_container_width=True)

with tab_qa:
    st.caption(
        "Ask a question in plain English. Gemini interprets it into a structured "
        "operation (group/filter/sort/describe/correlate) — it never runs arbitrary "
        "generated code against your data."
    )

    question = st.text_input("Your question", placeholder="e.g. What's the average value by category?")
    if st.button("Ask", type="primary") and question:
        try:
            with st.spinner("Thinking..."):
                result = answer_question(df, column_types, question)
            st.session_state.qa_history.insert(0, {"question": question, "result": result})
        except Exception as e:
            st.error(f"Couldn't answer that: {e}")

    for item in st.session_state.qa_history:
        with st.container(border=True):
            st.markdown(f"**Q: {item['question']}**")
            result = item["result"]
            if result.get("spec", {}).get("explanation"):
                st.caption(result["spec"]["explanation"])

            if result["type"] == "scalar":
                st.metric("Answer", result["value"])
            elif result["type"] == "table":
                st.dataframe(result["data"], use_container_width=True, hide_index=True)
            elif result["type"] == "explanation_only":
                st.info(result["explanation"])