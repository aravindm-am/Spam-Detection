import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from databricks import sql
import requests
import time

# Load Databricks secrets
DATABRICKS_HOST = st.secrets["databricks_host"]
DATABRICKS_PATH = st.secrets["databricks_http_path"]
DATABRICKS_TOKEN = st.secrets["databricks_token"]
DATABRICKS_NOTEBOOK_PATH = st.secrets["databricks_notebook_path"]

# Connect to Databricks
@st.cache_resource
def get_connection():
    try:
        conn = sql.connect(
            server_hostname=DATABRICKS_HOST,
            http_path=DATABRICKS_PATH,
            access_token=DATABRICKS_TOKEN
        )
        return conn
    except Exception as e:
        st.error(f"❌ Databricks connection failed: {e}")
        return None

# Check connection
conn = get_connection()
if conn:
    st.success("✅ Successfully connected to Databricks.")
else:
    st.stop()  # Do not continue if connection fails

# Function to run the notebook as a one-time job
def run_notebook(phone_number):
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    # Cluster ID of your existing cluster
    EXISTING_CLUSTER_ID = "0408-094007-uows7xsz"

    # Submit a new one-time job run
    submit_payload = {
        "run_name": f"FraudCheck_{phone_number}",
        "notebook_task": {
            "notebook_path": DATABRICKS_NOTEBOOK_PATH,
            "base_parameters": {
                "phone_number": phone_number
            }
        },
        "existing_cluster_id": EXISTING_CLUSTER_ID        
    }

    response = requests.post(
        f"{DATABRICKS_HOST}/api/2.1/jobs/runs/submit",
        headers=headers,
        json=submit_payload
    )

    if response.status_code != 200:
        st.error("❌ Failed to start Databricks job.")
        st.text(response.text)
        return None

    run_id = response.json()["run_id"]
    st.info(f"🚀 Job started (run_id={run_id}). Waiting for completion...")

    # Poll for status
    while True:
        status_response = requests.get(
            f"{DATABRICKS_INSTANCE}/api/2.1/jobs/runs/get?run_id={run_id}",
            headers=headers
        )
        run_state = status_response.json()["state"]["life_cycle_state"]
        if run_state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        time.sleep(5)

    result = status_response.json()
    notebook_output = result.get("notebook_output", {})
    return notebook_output.get("result", "✅ Job completed, but no output was returned.")

# Streamlit UI
st.title("📞 Telecom Fraud Detection")

phone_number = st.text_input("Enter Phone Number to Check")

if st.button("Run Fraud Check"):
    if phone_number.strip():
        with st.spinner("Running analysis on Databricks..."):
            output = run_notebook(phone_number.strip())
            st.success("🎉 Job finished!")
            st.code(output)
    else:
        st.warning("Please enter a phone number.")
