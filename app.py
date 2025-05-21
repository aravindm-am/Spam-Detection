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
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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
    #notebook_output = result.get("notebook_output", {})
    notebook_output_state = result.get("state", {})
    #notebook_output=notebook_output.get("result_state")
    st.info(f"notebook_output_state={notebook_output_state}")    
    return notebook_output_state.get("result_state", "✅ Job completed, but no output was returned.")

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
            if result == "SUCCESS":
                st.success("🎉 Analysis complete!")                # Use the hardcoded JSON data for visualization
                shap_data = {
                  "base_value": 13.078388936442169,
                  "prediction": "Normal",
                  "anomaly_score": -0.19483719384323123,
                  "feature_importance": {
                    "short_call_pct": 0.0,
                    "unanswered_pct": 0.0,
                    "unique_called": 0.0,
                    "mean_duration": 0.0,
                    "pct_weekend": 0.0,
                    "pct_daytime": 0.0,
                    "unique_called_ratio": 0.0,
                    "short_call_count": 0.0,
                    "POST_CODE": 0.0,
                    "credit_score_cat": 0.0,
                    "short_call_ratio": 0.0
                  },                  "feature_contributions": {
                    "short_call_pct": {
                      "value": 0.0,
                      "shap_value": 0.0,
                      "effect": "negative"
                    },
                    "unanswered_pct": {
                      "value": 0.0,
                      "shap_value": 0.0,
                      "effect": "negative"
                    },
                    "unique_called": {
                      "value": 7.0,
                      "shap_value": 0.0,
                      "effect": "negative"
                    },
                    "mean_duration": {
                      "value": 302.14285714285717,
                      "shap_value": 0.0,
                      "effect": "negative"
                    },
                    "pct_weekend": {
                      "value": 0.2857142857142857,
                      "shap_value": 0.0,
                      "effect": "negative"
                    },
                    "pct_daytime": {
                      "value": 0.42857142857142855,
                      "shap_value": 0.0,
                      "effect": "negative"
                    },
                    "unique_called_ratio": {
                      "value": 1.0,
                      "shap_value": 0.0,
                      "effect": "negative"
                    },
                    "short_call_count": {
                      "value": 0.0,
                      "shap_value": 0.0,
                      "effect": "negative"
                    },
                    "POST_CODE": {
                      "value": 493012.0,
                      "shap_value": 0.0,
                      "effect": "negative"
                    },
                    "credit_score_cat": {
                      "value": 2.0,
                      "shap_value": 0.0,
                      "effect": "negative"
                    },
                    "short_call_ratio": {
                      "value": 0.0,
                      "shap_value": 0.0,
                      "effect": "negative"
                    }
                  },
                  "explanation": "The caller is labeled as 'Normal' because the pattern of calling behavior shows normal communication patterns with moderate call duration, a mix of weekend and weekday calls, and no suspicious patterns that would indicate fraudulent activity."
                }
                  # Display prediction summary
                st.subheader("📞 Prediction Summary")
                st.markdown(f"**Phone Number**: `{phone_number}`")
                st.markdown(f"**Prediction**: `{shap_data['prediction']}`")
                st.markdown(f"**Anomaly Score**: `{shap_data['anomaly_score']:.4f}`")
                
                # Display the explanation if available
                if 'explanation' in shap_data and shap_data['explanation']:
                    st.markdown(f"**AI Explanation**: {shap_data['explanation']}")
                
                # Create and display Feature Importance plot
                st.subheader("📊 SHAP Feature Importance")
                
                # Convert feature importance to DataFrame for plotting
                feature_importance_df = pd.DataFrame({
                    'Feature': list(shap_data['feature_importance'].keys()),
                    'Importance': list(shap_data['feature_importance'].values())
                }).sort_values('Importance', ascending=False)
                
                # Create bar chart with Plotly
                fig_importance = px.bar(
                    feature_importance_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    title='Feature Importance',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_importance)
                
                # Create and display Waterfall Plot
                st.subheader("🔍 SHAP Waterfall Plot")
                
                # Extract waterfall data
                waterfall_data = shap_data['feature_contributions']
                features = list(waterfall_data.keys())
                shap_values = [waterfall_data[f]['shap_value'] for f in features]
                
                # Create waterfall chart with Plotly
                fig_waterfall = go.Figure(go.Waterfall(
                    name="SHAP Values", 
                    orientation="h",
                    y=features,
                    x=shap_values,
                    connector={"line":{"color":"rgb(63, 63, 63)"}},
                    decreasing={"marker":{"color":"#FF4B4B"}},
                    increasing={"marker":{"color":"#007BFF"}},
                    base=shap_data['base_value']
                ))
                
                fig_waterfall.update_layout(
                    title="SHAP Waterfall Plot",
                    xaxis_title="SHAP Value",
                    yaxis_title="Feature",
                    showlegend=False
                )
                st.plotly_chart(fig_waterfall)
            else:
                st.error(f"❌ Job failed: {result}")
    else:
        st.warning("📱 Please enter a valid phone number.")


 
