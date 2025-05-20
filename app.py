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
    
    # Try to get logs from the notebook run for debugging
    run_id = result.get("run_id")
    if run_id:
        try:
            logs_response = requests.get(
                f"{DATABRICKS_HOST}/api/2.0/jobs/runs/get-output",
                headers=headers,
                params={"run_id": run_id}
            )
            if logs_response.status_code == 200:
                logs_data = logs_response.json()
                st.info(f"Notebook logs retrieved: {len(logs_data.get('notebook_output', {}).get('result', ''))} characters")
        except Exception as e:
            st.warning(f"Could not retrieve logs: {e}")
    
    # Check if we have output data
    if not notebook_output or "result" not in notebook_output:
        # Check if there's any error information in the result
        if "error_message" in result.get("state", {}):
            return f"❌ Error in notebook: {result['state']['error_message']}"
        
        # Try to extract any useful info from the run state
        run_state_msg = result.get("state", {}).get("state_message", "")
        if run_state_msg:
            return f"ℹ️ {run_state_msg}"
            
        # Check if we can get any output details from the run page
        run_page_url = result.get("run_page_url")
        if run_page_url:
            st.info(f"You can check the run details at: {run_page_url}")
            
    return notebook_output.get("result", "✅ Job completed, but no output was returned.")

# Streamlit UI
st.title("📞 Telecom Fraud Detection")

phone_number = st.text_input("Enter Phone Number to Check")

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
                
                # Check for error message
                if result_data.get("error", False):
                    st.error(result_data.get("message", "Unknown error occurred"))
                    st.stop()
                
                st.success("🎉 Analysis complete!")
                
                # Display prediction summary
                st.subheader("📞 Prediction Summary")
                st.markdown(f"**Phone Number**: `{result_data['phone_number']}`")
                st.markdown(f"**Prediction**: `{result_data['prediction']}`")
                st.markdown(f"**Anomaly Score**: `{result_data['anomaly_score']:.4f}`")
                st.markdown(f"**Explanation**: {result_data['explanation']}")
                
                # Display top features (if available)
                if 'top_features' in result_data:
                    st.subheader("🔝 Top Features")
                    for i, feature in enumerate(result_data['top_features']):
                        if 'feature_values' in result_data and i < len(result_data['feature_values']):
                            st.markdown(f"- **{feature}**: `{result_data['feature_values'][i]:.4f}`")
                        else:
                            st.markdown(f"- **{feature}**")
                
                # Check if this is a text-only response
                is_text_only = result_data.get('is_text_only', False)
                
                # Display SHAP Feature Importance plot (if available)
                if not is_text_only and 'feature_importance_b64' in result_data:
                    st.subheader("📊 SHAP Feature Importance")
                    try:
                        feature_img = BytesIO(base64.b64decode(result_data['feature_importance_b64']))
                        st.image(feature_img)
                    except Exception as e:
                        st.warning(f"⚠ Could not load feature importance plot: {e}")
                
                # Display SHAP Waterfall plot (if available)
                if not is_text_only and 'waterfall_b64' in result_data and not result_data.get('is_reduced', False):
                    st.subheader("🔍 SHAP Waterfall Plot")
                    try:
                        waterfall_img = BytesIO(base64.b64decode(result_data['waterfall_b64']))
                        st.image(waterfall_img)
                    except Exception as e:                   
                        st.warning(f"⚠ Could not load waterfall plot: {e}")
                
                # Display a note if we're showing reduced data
                if result_data.get('is_reduced', False):
                    st.info("ℹ️ Note: Some visualizations were simplified or omitted due to size constraints.")
                
            except (json.JSONDecodeError, KeyError) as e:
                # If it's not valid JSON or doesn't have the expected fields, show the result as is
                st.error(f"❌ Job failed or returned unexpected format: {e}")
                st.info("Response details (for debugging):")
                st.code(result)
                
                # Add options to debug or view the Databricks run
                if isinstance(result, dict) and "run_page_url" in result:
                    st.markdown(f"[View job details in Databricks]({result['run_page_url']})")
                    
                # Add inspection of the result
                st.subheader("Debugging Info")
                if isinstance(result, dict):
                    st.json(result)
                else:
                    st.code(result)
    else:
        st.warning("📱 Please enter a valid phone number.")


