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
            f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id={run_id}",
            headers=headers
        )
        st.info(f"status_response={status_response}")
        run_state = status_response.json()["state"]["life_cycle_state"]
        st.info(f"run_state={run_state}")    
        if run_state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        time.sleep(5)

    result = status_response.json()
    st.info(f"result={result}")
    notebook_output = result.get("notebook_output", {})
    st.info(f"notebook_output={notebook_output}")    
    return notebook_output.get("result", "✅ Job completed, but no output was returned.")

# Streamlit UI
st.title("📞 Telecom Fraud Detection")

phone_number = st.text_input("Enter Phone Number to Check")

# if st.button("Run Fraud Check"):
#     if phone_number.strip():
#         with st.spinner("Running analysis on Databricks..."):
#             output = run_notebook(phone_number.strip())
#             st.success("🎉 Job finished!")
#             st.code(output)
#     else:
#         st.warning("Please enter a phone number.")

if st.button("Run Fraud Check", key="run_check_button"):
    if phone_number.strip():
        with st.spinner("Running analysis on Databricks..."):
            result = run_notebook(phone_number.strip())
            
            # Check if the result could be a JSON string
            try:
                import json
                import base64
                from io import BytesIO
                
                # Try to parse as JSON
                result_data = json.loads(result)
                st.success("🎉 Analysis complete!")
                
                # Display prediction summary
                st.subheader("📞 Prediction Summary")
                st.markdown(f"**Phone Number**: `{result_data['phone_number']}`")
                st.markdown(f"**Prediction**: `{result_data['prediction']}`")
                st.markdown(f"**Anomaly Score**: `{result_data['anomaly_score']:.4f}`")
                st.markdown(f"**Explanation**: {result_data['explanation']}")
                
                # Display SHAP Feature Importance plot
                st.subheader("📊 SHAP Feature Importance")
                try:
                    feature_img = BytesIO(base64.b64decode(result_data['feature_importance_b64']))
                    st.image(feature_img)
                except Exception as e:
                    st.warning(f"⚠ Could not load feature importance plot: {e}")
                
                # Display SHAP Waterfall plot
                st.subheader("🔍 SHAP Waterfall Plot")
                try:
                    waterfall_img = BytesIO(base64.b64decode(result_data['waterfall_b64']))
                    st.image(waterfall_img)
                except Exception as e:
                    st.warning(f"⚠ Could not load waterfall plot: {e}")
                    
            except (json.JSONDecodeError, KeyError) as e:
                # If it's not valid JSON or doesn't have the expected fields, show the result as is
                st.error(f"❌ Job failed or returned unexpected format: {result}")
                st.info("Response details (for debugging):")
                st.code(result)
            else:
                st.error(f"❌ Job failed: {result}")
    else:
        st.warning("📱 Please enter a valid phone number.")


