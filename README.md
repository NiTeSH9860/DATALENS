# DataLens

DataLens is a Streamlit dashboard for exploring CSV datasets. Users can create an account, upload a CSV, inspect its profile, view automatic visualizations, ask questions in plain English, and save datasets for later use.

## Features

- Postgres-backed registration and login
- Robust CSV loading and data cleaning
- Dataset profiling, missing-value summaries, and outlier detection
- Automatic Plotly visualizations
- Gemini-powered questions using validated data operations
- Per-user dataset storage in Postgres

## Requirements

- Python 3.10 or newer
- A PostgreSQL database
- A Gemini API key

## Local setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Create a local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

4. Set `DATABASE_URL`, `GEMINI_API_KEY`, and the SMTP variables in `.env`. Use an email provider app password where required; do not use your normal email password.

5. Start the app:

   ```powershell
   streamlit run app.py
   ```

The database tables are created automatically on the first run.

## Streamlit deployment

Add `DATABASE_URL`, `GEMINI_API_KEY`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, and `SMTP_FROM_EMAIL` under the deployment platform's secrets settings. Install dependencies from `requirements.txt`, then use `streamlit run app.py` as the start command.

Never commit `.env` or API keys to source control.

## Typical workflow

1. Register or log in.
2. Upload a CSV from the sidebar.
3. Review the Overview and Visualizations tabs.
4. Ask questions in the Ask Your Data tab.
5. Save the dataset to reload it later.
