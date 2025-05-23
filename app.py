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

# Information about the system
with st.expander("ℹ️ About this system"):
    st.markdown("""
    This telecom fraud detection system uses machine learning to:
    1. Analyze call patterns for individual phone numbers
    2. Compare patterns against the entire dataset
    3. Identify anomalous behavior that may indicate fraud
    4. Provide detailed visualizations for both individual and aggregate data
    
    The system now returns all visualization data directly in JSON format for dynamic rendering in the UI.
    """)

# Run detection when button clicked
if st.button("Run Fraud Check", key="run_check_button"):
    if phone_number.strip():
        with st.spinner("Subex Spam Scoring started in Databricks..."):
            result, notebook_output = run_notebook(phone_number.strip())
            
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
                
                # Display Combined Analysis section with data from JSON
                if 'combined_analysis' in notebook_output and notebook_output['combined_analysis']['status'] == 'success':
                    combined_data = notebook_output['combined_analysis']
                    
                    st.markdown("---")
                    st.header("🌐 Combined Dataset Analysis")
                    
                    # Create metrics row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", combined_data['total_records'])
                    with col2:
                        st.metric("Anomalies", f"{combined_data['anomaly_count']} ({combined_data['anomaly_percentage']:.1f}%)")
                    with col3:
                        st.metric("Normal Records", combined_data['normal_count'])
                    
                    # Create tabs for different visualizations from JSON data
                    combined_tabs = st.tabs(["Feature Importance", "Feature Impact", "Anomaly Distribution", "Correlation Heatmap"])
                    
                    # Check if 'visualizations' key exists in the data
                    if 'visualizations' in combined_data:
                        viz_data = combined_data['visualizations']
                        
                        with combined_tabs[0]:
                            st.subheader("Combined Feature Importance")
                            st.markdown("This chart shows the most important features across all records in the dataset.")
                            
                            # Create feature importance plot using Plotly from JSON data
                            if 'feature_importance' in viz_data:
                                # Convert feature importance data to DataFrame for plotting
                                fi_df = pd.DataFrame(viz_data['feature_importance']).sort_values('importance', ascending=False)
                                
                                fig_importance = px.bar(
                                    fi_df, 
                                    x='importance', 
                                    y='feature', 
                                    orientation='h',
                                    title='Feature Importance',
                                    color='importance',
                                    color_continuous_scale='Blues'
                                )
                                fig_importance.update_layout(get_plotly_layout("Combined Feature Importance"))
                                st.plotly_chart(fig_importance, use_container_width=True)
                                
                                # Show top features list
                                st.markdown("**Top 5 Most Important Features**")
                                for i, feature in enumerate(combined_data['top_features']):
                                    st.markdown(f"{i+1}. **{feature}**")
                            else:
                                st.warning("Feature importance data not found in the response.")
                        
                        with combined_tabs[1]:
                            st.subheader("Feature Value Impact")
                            st.markdown("This chart shows how different feature values impact the model's decision across the dataset.")
                            
                            # Create feature impact visualization from JSON data
                            if 'feature_impact' in viz_data:
                                # Select top features based on importance for clarity
                                top_features = combined_data['top_features'][:5]  # Top 5 features
                                
                                # Create a figure for SHAP summary plot using Plotly
                                fig = go.Figure()
                                
                                # For each top feature, create a scatter plot
                                for feature in top_features:
                                    if feature in viz_data['feature_impact']:
                                        feature_data = viz_data['feature_impact'][feature]
                                        
                                        fig.add_trace(go.Scatter(
                                            x=feature_data['feature_values'],
                                            y=feature_data['shap_values'],
                                            mode='markers',
                                            name=feature,
                                            marker=dict(
                                                size=8,
                                                opacity=0.7,
                                                line=dict(width=1)
                                            )
                                        ))
                                
                                fig.update_layout(get_plotly_layout("SHAP Values vs. Feature Values"))
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.warning("Feature impact data not found in the response.")
                        
                        with combined_tabs[2]:
                            st.subheader("Anomaly Score Distribution")
                            st.markdown("Distribution of anomaly scores across all records, with current number highlighted.")
                            
                            # Create anomaly distribution histogram from JSON data
                            if 'anomaly_distribution' in viz_data:
                                dist_data = viz_data['anomaly_distribution']
                                
                                # Create histogram using plotly
                                fig = go.Figure()
                                
                                # Add histogram bars
                                bin_centers = [(dist_data['bin_edges'][i] + dist_data['bin_edges'][i+1])/2 for i in range(len(dist_data['bin_edges'])-1)]
                                fig.add_trace(go.Bar(
                                    x=bin_centers,
                                    y=dist_data['histogram_values'],
                                    name='Anomaly Scores'
                                ))
                                
                                # Add threshold line
                                fig.add_shape(
                                    type="line",
                                    x0=dist_data['threshold'], 
                                    y0=0,
                                    x1=dist_data['threshold'], 
                                    y1=max(dist_data['histogram_values']),
                                    line=dict(color="red", width=2, dash="dash")
                                )
                                
                                # Add current phone number line if available
                                if dist_data['current_phone_score'] is not None:
                                    fig.add_shape(
                                        type="line",
                                        x0=dist_data['current_phone_score'], 
                                        y0=0,
                                        x1=dist_data['current_phone_score'], 
                                        y1=max(dist_data['histogram_values']),
                                        line=dict(color="green", width=2)
                                    )
                                    
                                    # Add annotation for current phone
                                    fig.add_annotation(
                                        x=dist_data['current_phone_score'],
                                        y=max(dist_data['histogram_values'])/2,
                                        text=f"Current: {dist_data['current_phone_number']}",
                                        showarrow=True,
                                        arrowhead=1
                                    )
                                
                                fig.update_layout(get_plotly_layout("Distribution of Anomaly Scores"))
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display percentile information
                                if 'anomaly_metrics' in combined_data:
                                    metrics = combined_data['anomaly_metrics']
                                    
                                    st.subheader("Anomaly Score Percentiles")
                                    percentiles_df = pd.DataFrame({
                                        'Percentile': ['25th', '50th (Median)', '75th', '90th', '99th'],
                                        'Score': [
                                            metrics['anomaly_score_percentiles']['25th'], 
                                            metrics['anomaly_score_percentiles']['50th'],
                                            metrics['anomaly_score_percentiles']['75th'],
                                            metrics['anomaly_score_percentiles']['90th'],
                                            metrics['anomaly_score_percentiles']['99th']
                                        ]
                                    })
                                    st.table(percentiles_df)
                                    
                                    # Show where this phone number's score falls in the distribution
                                    if 'anomaly_score' in notebook_output:
                                        current_score = notebook_output['anomaly_score']
                                        percentile_position = None
                                        
                                        if current_score <= metrics['anomaly_score_percentiles']['25th']:
                                            percentile_position = "below the 25th percentile"
                                        elif current_score <= metrics['anomaly_score_percentiles']['50th']:
                                            percentile_position = "between the 25th and 50th percentiles"
                                        elif current_score <= metrics['anomaly_score_percentiles']['75th']:
                                            percentile_position = "between the 50th and 75th percentiles"
                                        elif current_score <= metrics['anomaly_score_percentiles']['90th']:
                                            percentile_position = "between the 75th and 90th percentiles"
                                        elif current_score <= metrics['anomaly_score_percentiles']['99th']:
                                            percentile_position = "between the 90th and 99th percentiles"
                                        else:
                                            percentile_position = "above the 99th percentile"
                                        
                                        st.markdown(f"This phone number's anomaly score of **{current_score:.4f}** falls {percentile_position} of all scores.")
                            else:
                                st.warning("Anomaly distribution data not found in the response.")
                        
                        with combined_tabs[3]:
                            st.subheader("Feature Correlation Heatmap")
                            st.markdown("This heatmap shows correlations between features in the dataset.")
                            
                            # Create correlation heatmap from JSON data
                            if 'correlation_matrix' in viz_data:
                                # Convert the nested dictionary to a DataFrame
                                corr_data = pd.DataFrame(viz_data['correlation_matrix'])
                                
                                # Create heatmap using plotly
                                fig = px.imshow(
                                    corr_data,
                                    color_continuous_scale='RdBu_r',  # Blue (positive) to Red (negative)
                                    zmin=-1, zmax=1,  # Correlation range
                                )
                                
                                fig.update_layout(
                                    height=600,
                                    xaxis=dict(side="bottom"),
                                    title="Feature Correlation Matrix"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                st.markdown("Strong positive correlations appear in dark blue, while strong negative correlations appear in dark red.")
                            else:
                                st.warning("Correlation matrix data not found in the response.")
                    else:
                        st.warning("Visualization data not found in the response. The backend may be using an older version.")
                
            elif "error" in notebook_output:
                st.error(f"❌ Error: {notebook_output['error']}")
            else:
                st.error(f"❌ Job failed: {result}")
    else:
        st.warning("📱 Please enter a valid phone number.")
