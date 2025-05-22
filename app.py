import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import requests
import numpy as np
from databricks import sql

# Cache configuration values with increased TTL
@st.cache_data(ttl=600)  # 10 minutes cache for config
def get_config():
    return {
        "DATABRICKS_HOST": st.secrets["databricks_host"],
        "DATABRICKS_PATH": st.secrets["databricks_http_path"],
        "DATABRICKS_TOKEN": st.secrets["databricks_token"],
        "DATABRICKS_NOTEBOOK_PATH": st.secrets["databricks_notebook_path"],
        "EXISTING_CLUSTER_ID": "0521-131856-gsh3b6se"
    }

# Connect to Databricks - cached connection with increased TTL
@st.cache_resource(ttl=3600)  # 1 hour cache for connections
def get_connection():
    config = get_config()
    try:
        conn = sql.connect(
            server_hostname=config["DATABRICKS_HOST"],
            http_path=config["DATABRICKS_PATH"],
            access_token=config["DATABRICKS_TOKEN"]
        )
        return conn
    except Exception as e:
        st.error(f"❌ Databricks connection failed: {e}")
        return None

# Optimized function to run notebook with caching for repeated checks
@st.cache_data(ttl=300)  # Cache results for 5 minutes
def run_notebook(phone_number):
    config = get_config()
    headers = {
        "Authorization": f"Bearer {config['DATABRICKS_TOKEN']}",
        "Content-Type": "application/json"
    }

    # Submit job with optimized parameters
    submit_payload = {
        "run_name": f"FraudCheck_{phone_number}",
        "notebook_task": {
            "notebook_path": config["DATABRICKS_NOTEBOOK_PATH"],
            "base_parameters": {
                "phone_number": phone_number
            }
        },
        "existing_cluster_id": config["EXISTING_CLUSTER_ID"]
    }

    response = requests.post(
        f"{config['DATABRICKS_HOST']}/api/2.1/jobs/runs/submit",
        headers=headers,
        json=submit_payload,
        timeout=10  # Add timeout for faster failure detection
    )

    if response.status_code != 200:
        return "FAILED", {"error": response.text}
        
    run_id = response.json()["run_id"]
    
    # Removed status placeholder to eliminate the second spinner
    
    # More efficient polling with increasing backoff
    backoff = 1
    max_backoff = 5
    max_attempts = 20  # Prevent infinite loops
    attempts = 0
    
    while attempts < max_attempts:
        try:
            status_response = requests.get(
                f"{config['DATABRICKS_HOST']}/api/2.1/jobs/runs/get?run_id={run_id}",
                headers=headers,
                timeout=5  # Add timeout for faster polling
            )
            
            run_state = status_response.json()["state"]["life_cycle_state"]
            
            if run_state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
                break
                
            # Exponential backoff with cap
            time.sleep(min(backoff, max_backoff))
            backoff *= 1.5
            attempts += 1
        except requests.exceptions.Timeout:
            # Continue if timeout occurs during polling
            attempts += 1
            continue
    
    result = status_response.json()
    notebook_output_state = result.get("state", {})
    result_state = notebook_output_state.get("result_state", "UNKNOWN")
    
    # Get notebook output
    notebook_output = None
    if result_state == "SUCCESS":
        try:
            output_response = requests.get(
                f"{config['DATABRICKS_HOST']}/api/2.1/jobs/runs/get-output?run_id={run_id}",
                headers=headers,
                timeout=5
            )
            
            if output_response.status_code == 200:
                notebook_result = output_response.json().get("notebook_output", {})
                notebook_output = notebook_result.get("result", None)
                
                # Parse JSON output
                if isinstance(notebook_output, str):
                    try:
                        notebook_output = json.loads(notebook_output)
                    except:
                        notebook_output = {"error": "Failed to parse notebook output"}
        except requests.exceptions.Timeout:
            # Handle timeout in output retrieval
            notebook_output = {"error": "Timed out while retrieving notebook output"}
    
    return result_state, notebook_output

# Streamlit UI
st.title("📞 Telecom Fraud Detection")

# Cache the default layouts for faster rendering
@st.cache_data(ttl=3600)  # Cache layouts for 1 hour
def get_plotly_layout(title):
    return {
        "title": title,
        "showlegend": False,
        "height": 500,
        "margin": {"l": 40, "r": 40, "t": 50, "b": 40}
    }

# Main phone number input
phone_number = st.text_input("Enter Phone Number to Check")

# Run detection when button clicked
if st.button("Run Fraud Check", key="run_check_button"):
    if phone_number.strip():
        with st.spinner("Subex Spam Scoring started in Databricks..."):
            result, notebook_output = run_notebook(phone_number.strip())
            
            # Fixed error handling: Check if notebook_output is a dictionary and successful
            if result == "SUCCESS" and isinstance(notebook_output, dict) and notebook_output:
                st.success("🎉 Analysis complete!")
                
                # Display prediction summary
                st.subheader("📞 Prediction Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Phone Number**: `{phone_number}`")
                    st.markdown(f"**Prediction**: `{notebook_output.get('prediction', 'Unknown')}`")
                
                with col2:
                    anomaly_score = notebook_output.get('anomaly_score', 0)
                    time_taken = notebook_output.get('time_taken_seconds', 0)
                    st.markdown(f"**Anomaly Score**: `{anomaly_score:.4f}`")
                    st.markdown(f"**Processing Time**: `{time_taken:.2f}s`")
                
                # Display the explanation if available
                if 'explanation' in notebook_output and notebook_output['explanation']:
                    st.markdown(f"**AI Explanation**: {notebook_output['explanation']}")
                
                # Performance metrics if available
                if 'performance_metrics' in notebook_output:
                    with st.expander("Performance Metrics"):
                        metrics = notebook_output['performance_metrics']
                        st.write(f"📊 Data Loading: {metrics.get('load_time', 0):.2f}s")
                        st.write(f"📊 Data Processing: {metrics.get('process_time', 0):.2f}s")
                        st.write(f"📊 Model Inference: {metrics.get('model_time', 0):.2f}s")
                        st.write(f"📊 API Processing: {metrics.get('api_time', 0):.2f}s")
                
                # Tabs for different visualizations
                tab1, tab2 = st.tabs(["Feature Importance", "SHAP Waterfall"])
                
                with tab1:
                    # Safely get feature importance data
                    if 'feature_importance' in notebook_output:
                        # Convert feature importance to DataFrame for plotting
                        feature_importance_df = pd.DataFrame({
                            'Feature': list(notebook_output['feature_importance'].keys()),
                            'Importance': list(notebook_output['feature_importance'].values())
                        }).sort_values('Importance', ascending=False)
                        
                        # Create bar chart with Plotly
                        fig_importance = px.bar(
                            feature_importance_df, 
                            x='Importance', 
                            y='Feature', 
                            orientation='h',
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        fig_importance.update_layout(get_plotly_layout("Feature Importance"))
                        st.plotly_chart(fig_importance, use_container_width=True)
                    else:
                        st.warning("Feature importance data not available")
                
                with tab2:
                    # Safely extract waterfall data
                    if 'feature_contributions' in notebook_output and 'base_value' in notebook_output:
                        waterfall_data = notebook_output['feature_contributions']
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
                            base=notebook_output['base_value']
                        ))
                        
                        fig_waterfall.update_layout(get_plotly_layout("SHAP Waterfall Plot"))
                        st.plotly_chart(fig_waterfall, use_container_width=True)
                    else:
                        st.warning("SHAP waterfall data not available")
                    
            # Fixed error handling: Check if notebook_output is a dictionary before accessing 'error'
            elif isinstance(notebook_output, dict) and "error" in notebook_output:
                st.error(f"❌ Error: {notebook_output['error']}")
            else:
                st.error(f"❌ Job failed: {result}")
    else:
        st.warning("📱 Please enter a valid phone number.")
