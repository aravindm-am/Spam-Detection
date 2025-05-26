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
    EXISTING_CLUSTER_ID = "0521-131856-gsh3b6se"

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
    
    # Create a status placeholder for the user-friendly message
    status_placeholder = st.empty()
    # status_placeholder.info("🔍 Subex Spam Scoring Started in Databricks...")

    # Poll for status silently (without showing technical details)
    while True:
        status_response = requests.get(
            f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id={run_id}",
            headers=headers
        )
        
        # Get state but don't display technical messages
        run_state = status_response.json()["state"]["life_cycle_state"]
        
        if run_state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        time.sleep(1)
      # Clear the status message when done
    status_placeholder.empty()

    # result = status_response.json()
    # # Removed debug info messages


    
    # notebook_output_state = result.get("state", {})
    # return notebook_output_state.get("result_state", "✅ Job completed, but no output was returned.")
    result = status_response.json()
    # st.info(f"result ==== {result}")
    notebook_output_state = result.get("state", {})
    result_state = notebook_output_state.get("result_state", "UNKNOWN")
    
    # Get notebook output if available
    notebook_output = None
    if result_state == "SUCCESS":
        # Get notebook output from job result
        output_response = requests.get(
            f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get-output?run_id={run_id}",
            headers=headers
        )
        # st.info(f"output_response ==== {output_response.json()}")
        
        if output_response.status_code == 200:
            notebook_result = output_response.json().get("notebook_output", {})
            # st.info(f"output_response ==== {notebook_result}")
            # The result might be in result.data or result.result
            notebook_output = notebook_result.get("result", None)
            
            # Try to parse the output as JSON if it's a string
            if isinstance(notebook_output, str):
                try:
                    notebook_output = json.loads(notebook_output)
                except:
                    pass  # Keep as string if not valid JSON
    
    return result_state, notebook_output

# Streamlit UI
st.title("📞 Telecom Fraud Detection")

phone_number = st.text_input("Enter Phone Number to Check")

# if st.button("Run Fraud Check"):
#     if phone_number.strip():
#         with st.spinner("Subex Spam Scoring Started in Databricks..."):
#             output = run_notebook(phone_number.strip())
#             st.success("🎉 Job finished!")
#             st.code(output)
#     else:
#         st.warning("Please enter a phone number.")

if st.button("Run Fraud Check", key="run_check_button"):
    if phone_number.strip():
        with st.spinner("Subex Spam Scoring Started in Databricks..."):
            result, notebook_output = run_notebook(phone_number.strip())
            if result == "SUCCESS":
                st.success("🎉 Analysis complete!")                # Use the hardcoded JSON data for visualization
                shap_data = {
                  "base_value": 10.546970406720215,
                  "prediction": "Anomaly",
                  "anomaly_score": 0.012908768548510419,
                  "feature_importance": {
                    "mean_duration": 1.762012160930317,
                    "pct_daytime": 1.702906182524166,
                    "pct_weekend": 0.16904060471395496,
                    "POST_CODE": 0.16689157510933,
                    "credit_score_cat": 0.03107703131856397,
                    "short_call_pct": 0.0,
                    "unanswered_pct": 0.0,
                    "unique_called": 0.0,
                    "unique_called_ratio": 0.0,
                    "short_call_count": 0.0,
                    "short_call_ratio": 0.0
                  },
                  "feature_contributions": {
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
                      "value": 268.85714285714283,
                      "shap_value": -1.762012160930317,
                      "effect": "negative"
                    },
                    "pct_weekend": {
                      "value": 0.14285714285714285,
                      "shap_value": 0.16904060471395496,
                      "effect": "positive"
                    },
                    "pct_daytime": {
                      "value": 0.8571428571428571,
                      "shap_value": -1.702906182524166,
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
                      "value": 879945.0,
                      "shap_value": -0.16689157510933,
                      "effect": "negative"
                    },
                    "credit_score_cat": {
                      "value": 3.0,
                      "shap_value": -0.03107703131856397,
                      "effect": "negative"
                    },
                    "short_call_ratio": {
                      "value": 0.0,
                      "shap_value": 0.0,
                      "effect": "negative"
                    }
                  },
                  "explanation": "Caller 917267973248 is labeled as an 'Anomaly' in the telecom fraud detection system due to having a high unique_called value of 7.0, which is unusual compared to normal calling patterns and may indicate suspicious behavior."
                }
                shap_data = notebook_output
                  # Display prediction summary
                st.subheader("📞 Prediction Summary")
                st.markdown(f"**Phone Number**: `{phone_number}`")
                st.markdown(f"**Prediction**: `{shap_data['prediction']}`")
                st.markdown(f"**Anomaly Score**: `{shap_data['anomaly_score']:.4f}`")                # Display the explanation if available
                if 'explanation' in shap_data and shap_data['explanation']:
                    st.markdown(f"**AI Explanation**: {shap_data['explanation']}")
                
                # Create main tabs for different analysis views
                analysis_tab1, analysis_tab2 = st.tabs(["🔎 Individual Analysis", "📈 Combined Analysis"])
                
                # Prepare data for visualizations
                feature_importance_df = pd.DataFrame({
                    'Feature': list(shap_data['feature_importance'].keys()),
                    'Importance': list(shap_data['feature_importance'].values())
                }).sort_values('Importance', ascending=False)
                
                waterfall_data = shap_data['feature_contributions']
                features = list(waterfall_data.keys())
                shap_values = [waterfall_data[f]['shap_value'] for f in features]
                
                # Tab 1: Individual Analysis
                with analysis_tab1:
                    # Create subtabs for Feature Importance and Waterfall Plot
                    tab1, tab2 = st.tabs(["📊 Feature Importance", "🔍 Waterfall"])
                    
                    # Subtab 1: Feature Importance
                    with tab1:
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
                    
                    # Subtab 2: Waterfall Plot
                    with tab2:
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
                        st.plotly_chart(fig_waterfall)                  # Tab 2: Combined Analysis
                with analysis_tab2:
                    if 'combined_analysis' in shap_data:
                        combined = shap_data['combined_analysis']
                        
                        # Create tabs for different combined analyses
                        combined_tab1, combined_tab2, combined_tab3, combined_tab4 = st.tabs([
                            "📊 Global Feature Importance", 
                            "📈 Feature Distribution", 
                            "🔄 Feature Correlation", 
                            "🔔 Anomaly Distribution"
                        ])
                        
                        # Tab 1: Global Feature Importance
                        with combined_tab1:
                            st.markdown("### Global SHAP Feature Importance")
                            st.markdown("This shows the average impact of each feature across all samples in the dataset.")
                            
                            # Convert global feature importance to DataFrame
                            if 'global_feature_importance' in combined:
                                global_importance_df = pd.DataFrame({
                                    'Feature': list(combined['global_feature_importance'].keys()),
                                    'Importance': list(combined['global_feature_importance'].values())
                                }).sort_values('Importance', ascending=False)
                                
                                # Create bar chart with Plotly
                                fig_global_importance = px.bar(
                                    global_importance_df, 
                                    x='Importance', 
                                    y='Feature', 
                                    orientation='h',
                                    color='Importance',
                                    color_continuous_scale='Viridis',
                                    title='Global Feature Importance (All Data)'
                                )
                                st.plotly_chart(fig_global_importance, use_container_width=True)
                            else:
                                st.warning("Global feature importance data not available.")
                        
                        # Tab 2: Feature Distribution by Prediction
                        with combined_tab2:
                            st.markdown("### Feature Distribution: Normal vs Anomaly")
                            st.markdown("Compare how feature values differ between normal and anomalous calls.")
                            
                            if 'feature_distributions' in combined:
                                # Let user select which feature to examine
                                select_feature = st.selectbox(
                                    "Select feature to analyze:", 
                                    options=list(combined['feature_distributions'].keys())
                                )
                                
                                if select_feature:
                                    feature_dist = combined['feature_distributions'][select_feature]
                                    
                                    # Create a comparison dataframe
                                    normal_values = feature_dist['normal']
                                    anomaly_values = feature_dist['anomaly']
                                    
                                    comparison_df = pd.DataFrame({
                                        'Statistic': list(normal_values.keys()),
                                        'Normal': list(normal_values.values()),
                                        'Anomaly': list(anomaly_values.values())
                                    })
                                    
                                    # Display as a table
                                    st.dataframe(comparison_df)
                                    
                                    # Create visualization
                                    fig_dist = go.Figure()
                                    fig_dist.add_trace(go.Bar(
                                        x=list(normal_values.keys())[1:-1],  # Skip count and first percentile
                                        y=list(normal_values.values())[1:-1],
                                        name="Normal",
                                        marker_color='#007BFF'
                                    ))
                                    fig_dist.add_trace(go.Bar(
                                        x=list(anomaly_values.keys())[1:-1],  # Skip count and first percentile 
                                        y=list(anomaly_values.values())[1:-1],
                                        name="Anomaly",
                                        marker_color='#FF4B4B'
                                    ))
                                    
                                    fig_dist.update_layout(
                                        title=f"Distribution Statistics: {select_feature}",
                                        xaxis_title="Statistic",
                                        yaxis_title="Value",
                                        barmode='group'
                                    )
                                    
                                    st.plotly_chart(fig_dist, use_container_width=True)
                            else:
                                st.warning("Feature distribution data not available.")
                        
                        # Tab 3: Correlation Matrix
                        with combined_tab3:
                            st.markdown("### Feature Correlation Matrix")
                            st.markdown("See how features correlate with each other across all samples.")
                            
                            if 'correlation_matrix' in combined:
                                # Convert correlation matrix to DataFrame
                                corr_df = pd.DataFrame.from_dict(combined['correlation_matrix'])
                                
                                # Create heatmap
                                fig_corr = px.imshow(
                                    corr_df,
                                    color_continuous_scale='RdBu_r',
                                    zmin=-1, 
                                    zmax=1,
                                    title='Feature Correlation Matrix'
                                )
                                
                                fig_corr.update_layout(
                                    height=600,
                                    width=700
                                )
                                
                                st.plotly_chart(fig_corr, use_container_width=True)
                            else:
                                st.warning("Correlation matrix data not available.")
                        
                        # Tab 4: Anomaly Score Distribution
                        with combined_tab4:
                            st.markdown("### Anomaly Score Distribution")
                            st.markdown("Distribution of anomaly scores for normal and anomalous samples.")
                            
                            if 'anomaly_score_distribution' in combined and 'prediction_distribution' in combined:
                                # Create a pie chart for prediction distribution
                                labels = list(combined['prediction_distribution'].keys())
                                values = list(combined['prediction_distribution'].values())
                                
                                fig_pie = px.pie(
                                    names=labels,
                                    values=values,
                                    title='Prediction Distribution',
                                    color=labels,
                                    color_discrete_map={'Normal': '#007BFF', 'Anomaly': '#FF4B4B'}
                                )
                                
                                st.plotly_chart(fig_pie, use_container_width=True)
                                
                                # Create histogram for anomaly scores
                                hist_data = combined['anomaly_score_distribution']['histogram_data']
                                
                                fig_hist = go.Figure()
                                
                                # Convert bin edges to bin centers and labels
                                bins = hist_data['bins']
                                bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
                                bin_labels = [f"{bins[i]} to {bins[i+1]}" for i in range(len(bins)-1)]
                                
                                fig_hist.add_trace(go.Bar(
                                    x=bin_centers,
                                    y=hist_data['normal_counts'],
                                    name='Normal',
                                    marker_color='#007BFF',
                                    hovertemplate='Bin: %{text}<br>Count: %{y}<extra></extra>',
                                    text=bin_labels
                                ))
                                
                                fig_hist.add_trace(go.Bar(
                                    x=bin_centers,
                                    y=hist_data['anomaly_counts'],
                                    name='Anomaly',
                                    marker_color='#FF4B4B',
                                    hovertemplate='Bin: %{text}<br>Count: %{y}<extra></extra>',
                                    text=bin_labels
                                ))
                                
                                fig_hist.update_layout(
                                    title='Anomaly Score Distribution',
                                    xaxis_title='Anomaly Score',
                                    yaxis_title='Count',
                                    barmode='group'
                                )
                                
                                st.plotly_chart(fig_hist, use_container_width=True)
                            else:
                                st.warning("Anomaly score distribution data not available.")
                    else:
                        st.info("Combined analysis data is not available for this result.")
            else:
                st.error(f"❌ Job failed: {result}")
    else:
        st.warning("📱 Please enter a valid phone number.")


 
