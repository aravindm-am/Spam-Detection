import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from databricks import sql
import requests
import time
import json

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

    # result = status_response.json()
    # st.info(f"result={result}")
    # # notebook_output = result.get("notebook_output", {})
    # notebook_output_state = result.get("state", {})
    # # notebook_output=notebook_output.get("result_state")
    # st.info(f"notebook_output_state={notebook_output_state}")    
    
    result = status_response.json()
    st.info(f"result={result}")
    notebook_output = result.get("notebook_output", {})
    result_json = notebook_output.get("result", "{}")
 
    try:
        return json.loads(result_json)
    except Exception as e:
        st.error(f"❌ Failed to parse notebook output: {e}")
        return None

    # return notebook_output_state.get("result_state", "✅ Job completed, but no output was returned.")

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
            # if result == "SUCCESS":
            #     st.success("🎉 Analysis complete!")

            #     # Load prediction result
            #     try:
            #         result_df = pd.read_csv("/Workspace/Users/aravind.menon@subex.com/Spam Detection/sample_number_predictions.csv")
            #         row = result_df.iloc[0]
            #         st.subheader("📞 Prediction Summary")
            #         st.markdown(f"**Phone Number**: `{row['caller']}`")
            #         st.markdown(f"**Prediction**: `{row['prediction']}`")
            #         st.markdown(f"**Anomaly Score**: `{row['anomaly_score']:.4f}`")
            #         st.markdown(f"**Explanation**: {row['explanation']}")
            #     except Exception as e:
            #         st.error(f"❌ Failed to read prediction: {e}")

            #     # Load SHAP plots
            #     st.subheader("📊 SHAP Feature Importance")
            #     try:
            #         st.image("/Workspace/Users/aravind.menon@subex.com/Spam Detection/feature_importance.png")
            #     except Exception as e:
            #         st.warning(f"⚠ Could not load feature importance plot: {e}")

            #     st.subheader("🔍 SHAP Waterfall Plot")
            #     try:
            #         st.image("/Workspace/Users/aravind.menon@subex.com/Spam Detection/waterfall_plot.png")                    
            #     except Exception as e:
            #         st.warning(f"⚠ Could not load waterfall plot: {e}")

            if result and result.get("status") == "success":
                st.success("🎉 Analysis complete!")
             
                st.subheader("📞 Prediction Summary")
                st.markdown(f"**Phone Number**: `{result['caller']}`")
                st.markdown(f"**Prediction**: `{result['prediction']}`")
                st.markdown(f"**Anomaly Score**: `{result['anomaly_score']:.4f}`")
                st.markdown(f"**Explanation**: {result['explanation']}")
             
                # Render SHAP bar plot
                st.subheader("📊 SHAP Feature Importance")
                st.image(f"data:image/png;base64,{result['bar_plot_base64']}")
             
                # Render SHAP waterfall plot
                st.subheader("🔍 SHAP Waterfall Plot")
                st.image(f"data:image/png;base64,{result['waterfall_plot_base64']}")            
            else:
                st.error(f"❌ Job failed: {result}")
    else:
        st.warning("📱 Please enter a valid phone number.")
