import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import time
import requests
import numpy as np
from databricks import sql

# Cache configuration values
@st.cache_data
def get_config():
    return {
        "DATABRICKS_HOST": st.secrets["databricks_host"],
        "DATABRICKS_PATH": st.secrets["databricks_http_path"],
        "DATABRICKS_TOKEN": st.secrets["databricks_token"],
        "DATABRICKS_NOTEBOOK_PATH": st.secrets["databricks_notebook_path"],
        "EXISTING_CLUSTER_ID": "0521-131856-gsh3b6se"
    }

# Connect to Databricks - cached connection
@st.cache_resource
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
        json=submit_payload
    )

    if response.status_code != 200:
        return "FAILED", {"error": response.text}
        
    run_id = response.json()["run_id"]
    
    # Status placeholder
    status_placeholder = st.empty()
    
    # More efficient polling with increasing backoff
    backoff = 1
    max_backoff = 5
    while True:
        status_response = requests.get(
            f"{config['DATABRICKS_HOST']}/api/2.1/jobs/runs/get?run_id={run_id}",
            headers=headers
        )
        
        run_state = status_response.json()["state"]["life_cycle_state"]
        
        if run_state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
            
        # Exponential backoff with cap
        time.sleep(min(backoff, max_backoff))
        backoff *= 1.5
    
    status_placeholder.empty()

    result = status_response.json()
    notebook_output_state = result.get("state", {})
    result_state = notebook_output_state.get("result_state", "UNKNOWN")
    
    # Get notebook output
    notebook_output = None
    if result_state == "SUCCESS":
        output_response = requests.get(
            f"{config['DATABRICKS_HOST']}/api/2.1/jobs/runs/get-output?run_id={run_id}",
            headers=headers
        )
        
        if output_response.status_code == 200:
            notebook_result = output_response.json().get("notebook_output", {})
            notebook_output = notebook_result.get("result", None)
            
            # Parse JSON output
            if isinstance(notebook_output, str):
                try:
                    notebook_output = json.loads(notebook_output)
                except:
                    pass
    
    return result_state, notebook_output

# Streamlit UI
st.title("📞 Telecom Fraud Detection")

# Cache the default layouts for faster rendering
@st.cache_data
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
        st.spinner("Subex Spam Scoring started in Databricks..."):
            
            
            if result == "SUCCESS" and notebook_output:
                st.success("🎉 Analysis complete!")
                
                # Display prediction summary
                st.subheader("📞 Prediction Summary")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Phone Number**: `{phone_number}`")
                    st.markdown(f"**Prediction**: `{notebook_output['prediction']}`")
                
                with col2:
                    st.markdown(f"**Anomaly Score**: `{notebook_output['anomaly_score']:.4f}`")
                    st.markdown(f"**Processing Time**: `{notebook_output.get('time_taken_seconds', '?'):.2f}s`")
                
                # Display the explanation if available
                if 'explanation' in notebook_output and notebook_output['explanation']:
                    st.markdown(f"**AI Explanation**: {notebook_output['explanation']}")
                
                # Tabs for different visualizations
                tab1, tab2 = st.tabs(["Feature Importance", "SHAP Waterfall"])
                
                with tab1:
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
                
                with tab2:
                    # Extract waterfall data
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
                    
            elif "error" in notebook_output:
                st.error(f"❌ Error: {notebook_output['error']}")
            else:
                st.error(f"❌ Job failed: {result}")
    else:
        st.warning("📱 Please enter a valid phone number.")
