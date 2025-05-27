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

# Hardcoded combined analysis data to avoid running analysis every time
HARDCODED_COMBINED_ANALYSIS = {
    "global_feature_importance": {
        "short_call_ratio": 0.335,
        "unique_called_ratio": 0.287,
        "pct_daytime": 0.243,
        "mean_duration": 0.214,
        "pct_weekend": 0.180,
        "unanswered_pct": 0.163,
        "short_call_pct": 0.148,
        "credit_score_cat": 0.142,
        "PREPAYTYPE": 0.134,
        "PENALTY": 0.125,
        "POST_CODE": 0.108,
        "SUBSIDY": 0.097
    },
    "feature_distributions": {
        "short_call_ratio": {
            "normal": {
                "count": 9582.0,
                "mean": 0.183,
                "std": 0.109,
                "min": 0.0,
                "25%": 0.111,
                "50%": 0.167,
                "75%": 0.235,
                "max": 1.0
            },
            "anomaly": {
                "count": 942.0,
                "mean": 0.537,
                "std": 0.218,
                "min": 0.0,
                "25%": 0.375,
                "50%": 0.556,
                "75%": 0.714,
                "max": 1.0
            }
        },
        "unique_called_ratio": {
            "normal": {
                "count": 9582.0,
                "mean": 0.856,
                "std": 0.192,
                "min": 0.091,
                "25%": 0.75,
                "50%": 0.944,
                "75%": 1.0,
                "max": 1.0
            },
            "anomaly": {
                "count": 942.0,
                "mean": 0.424,
                "std": 0.205,
                "min": 0.062,
                "25%": 0.286,
                "50%": 0.4,
                "75%": 0.556,
                "max": 1.0
            }
        },
        "mean_duration": {
            "normal": {
                "count": 9582.0,
                "mean": 183.7,
                "std": 89.4,
                "min": 0.0,
                "25%": 127.8,
                "50%": 175.2,
                "75%": 230.5,
                "max": 596.2
            },
            "anomaly": {
                "count": 942.0,
                "mean": 42.6,
                "std": 31.9,
                "min": 0.0,
                "25%": 18.3,
                "50%": 35.1,
                "75%": 61.7,
                "max": 202.6
            }
        },
        "pct_daytime": {
            "normal": {
                "count": 9582.0,
                "mean": 0.651,
                "std": 0.232,
                "min": 0.0,
                "25%": 0.5,
                "50%": 0.667,
                "75%": 0.833,
                "max": 1.0
            },
            "anomaly": {
                "count": 942.0,
                "mean": 0.312,
                "std": 0.254,
                "min": 0.0,
                "25%": 0.091,
                "50%": 0.273,
                "75%": 0.5,
                "max": 1.0
            }
        },
        "pct_weekend": {
            "normal": {
                "count": 9582.0,
                "mean": 0.277,
                "std": 0.189,
                "min": 0.0,
                "25%": 0.143,
                "50%": 0.25,
                "75%": 0.4,
                "max": 1.0
            },
            "anomaly": {
                "count": 942.0,
                "mean": 0.523,
                "std": 0.258,
                "min": 0.0,
                "25%": 0.3,
                "50%": 0.545,
                "75%": 0.75,
                "max": 1.0
            }
        },
        "unanswered_pct": {
            "normal": {
                "count": 9582.0,
                "mean": 0.132,
                "std": 0.127,
                "min": 0.0,
                "25%": 0.0,
                "50%": 0.111,
                "75%": 0.2,
                "max": 0.909
            },
            "anomaly": {
                "count": 942.0,
                "mean": 0.289,
                "std": 0.224,
                "min": 0.0,
                "25%": 0.111,
                "50%": 0.25,
                "75%": 0.429,
                "max": 1.0
            }
        }
    },
    "correlation_matrix": {
        "short_call_ratio": {
            "short_call_ratio": 1.0,
            "unique_called_ratio": -0.472,
            "pct_daytime": -0.485,
            "mean_duration": -0.651,
            "pct_weekend": 0.412,
            "unanswered_pct": 0.338
        },
        "unique_called_ratio": {
            "short_call_ratio": -0.472,
            "unique_called_ratio": 1.0,
            "pct_daytime": 0.526,
            "mean_duration": 0.537,
            "pct_weekend": -0.352,
            "unanswered_pct": -0.312
        },
        "pct_daytime": {
            "short_call_ratio": -0.485,
            "unique_called_ratio": 0.526,
            "pct_daytime": 1.0,
            "mean_duration": 0.563,
            "pct_weekend": -0.489,
            "unanswered_pct": -0.377
        },
        "mean_duration": {
            "short_call_ratio": -0.651,
            "unique_called_ratio": 0.537,
            "pct_daytime": 0.563,
            "mean_duration": 1.0,
            "pct_weekend": -0.447,
            "unanswered_pct": -0.468
        },
        "pct_weekend": {
            "short_call_ratio": 0.412,
            "unique_called_ratio": -0.352,
            "pct_daytime": -0.489,
            "mean_duration": -0.447,
            "pct_weekend": 1.0,
            "unanswered_pct": 0.308
        },
        "unanswered_pct": {
            "short_call_ratio": 0.338,
            "unique_called_ratio": -0.312,
            "pct_daytime": -0.377,
            "mean_duration": -0.468,
            "pct_weekend": 0.308,
            "unanswered_pct": 1.0
        }
    },
    "anomaly_score_distribution": {
        "histogram_data": {
            "bins": [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "normal_counts": [0, 0, 0, 0, 0, 72, 845, 3128, 4362, 1175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "anomaly_counts": [0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 175, 356, 287, 79, 7, 0, 0, 0, 0, 0]
        },
        "statistics": {
            "normal": {
                "count": 9582.0,
                "mean": -2.547,
                "std": 0.783,
                "min": -5.872,
                "25%": -2.932,
                "50%": -2.421,
                "75%": -2.051,
                "max": -1.024
            },
            "anomaly": {
                "count": 942.0,
                "mean": 1.832,
                "std": 0.968,
                "min": 0.127,
                "25%": 1.087,
                "50%": 1.783,
                "75%": 2.376,
                "max": 4.721
            }
        }
    },
    "prediction_distribution": {
        "Normal": 9582,
        "Anomaly": 942
    }
}

# Load Databricks secrets
DATABRICKS_HOST = st.secrets["databricks_host"]
DATABRICKS_PATH = st.secrets["databricks_http_path"]
DATABRICKS_TOKEN = st.secrets["databricks_token"]
DATABRICKS_NOTEBOOK_PATH = st.secrets["databricks_notebook_path"]

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
    st.stop()

# Function to run the notebook job
def run_notebook(phone_number):
    headers = {
        "Authorization": f"Bearer {DATABRICKS_TOKEN}",
        "Content-Type": "application/json"
    }

    EXISTING_CLUSTER_ID = "0521-131856-gsh3b6se"

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
    status_placeholder = st.empty()

    while True:
        status_response = requests.get(
            f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id={run_id}",
            headers=headers
        )
        run_state = status_response.json()["state"]["life_cycle_state"]
        if run_state in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
            break
        time.sleep(1)

    status_placeholder.empty()
    result = status_response.json()
    result_state = result.get("state", {}).get("result_state", "UNKNOWN")
    
    notebook_output = None
    if result_state == "SUCCESS":
        output_response = requests.get(
            f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get-output?run_id={run_id}",
            headers=headers
        )
        if output_response.status_code == 200:
            notebook_result = output_response.json().get("notebook_output", {})
            notebook_output = notebook_result.get("result", None)
            if isinstance(notebook_output, str):
                try:
                    notebook_output = json.loads(notebook_output)
                except:
                    pass

    return result_state, notebook_output

# Streamlit UI
st.title("📞 Telecom Fraud Detection")
analysis_tab1, analysis_tab2 = st.tabs(["📈 Combined Analysis","🔎 Individual Analysis"])

# Tab 2
with analysis_tab2:
    phone_number = st.text_input("Enter Phone Number to Check")
    run_button = st.button("Run Fraud Check", key="run_check_button")
    
    if run_button:
        if phone_number.strip():
            with st.spinner("Subex Spam Scoring Started in Databricks..."):
                result, notebook_output = run_notebook(phone_number.strip())
                if result == "SUCCESS":
                    st.success("🎉 Analysis complete!")
                    shap_data = notebook_output
                    st.session_state.shap_data = shap_data

                    st.subheader("📞 Prediction Summary")
                    st.markdown(f"**Phone Number**: `{phone_number}`")
                    st.markdown(f"**Prediction**: `{shap_data['prediction']}`")
                    st.markdown(f"**Anomaly Score**: `{shap_data['anomaly_score']:.4f}`")

                    if 'explanation' in shap_data and shap_data['explanation']:
                        st.markdown(f"**AI Explanation**: {shap_data['explanation']}")

                    feature_importance_df = pd.DataFrame({
                        'Feature': list(shap_data['feature_importance'].keys()),
                        'Importance': list(shap_data['feature_importance'].values())
                    }).sort_values('Importance', ascending=False)

                    waterfall_data = shap_data['feature_contributions']
                    features = list(waterfall_data.keys())
                    shap_values = [waterfall_data[f]['shap_value'] for f in features]

                    tab1, tab2 = st.tabs(["📊 Feature Importance", "🔍 Waterfall"])

                    with tab1:
                        st.markdown("### 📊 Individual Feature Importance")
                        fig_importance = px.bar(
                            feature_importance_df, 
                            x='Importance', 
                            y='Feature', 
                            orientation='h',
                            title='Individual Feature Importance',
                            color='Importance',
                            color_continuous_scale='Blues'
                        )
                        st.plotly_chart(fig_importance)

                    with tab2:
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

# Tab 2
with analysis_tab1:
    # Check if we have a real analysis or should use the hardcoded data
    if 'shap_data' in st.session_state and 'combined_analysis' in st.session_state.shap_data:
        shap_data = st.session_state.shap_data
        combined = shap_data['combined_analysis']
        st.success("✅ Displaying analysis from the latest run")
    else:
        # Use hardcoded combined analysis
        combined = HARDCODED_COMBINED_ANALYSIS
        st.info("ℹ️ Displaying pre-computed analysis. Run an individual analysis for real-time data.")

        # Create a container for all visualizations with custom CSS to control the height
        with st.container():
            # Set up CSS to make the container take up the full viewport height
            st.markdown("""
                <style>
                .stPlotlyChart {
                    margin-bottom: 0 !important;
                }
                .main .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Create a three-column layout for the top row
            col1, col2, col3 = st.columns([2, 1, 1])
            
            # Column 1: Global Feature Importance
            with col1:
                if 'global_feature_importance' in combined:
                    st.markdown("#### 📊 Global SHAP Feature Importance")
                    global_importance_df = pd.DataFrame({
                        'Feature': list(combined['global_feature_importance'].keys()),
                        'Importance': list(combined['global_feature_importance'].values())
                    }).sort_values('Importance', ascending=False)
                    
                    # Show only top 10 features for better visibility
                    global_importance_df = global_importance_df.head(10)
                    
                    fig_global_importance = px.bar(
                        global_importance_df, 
                        x='Importance', 
                        y='Feature', 
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Viridis',
                        title=None
                    )
                    fig_global_importance.update_layout(
                        height=330,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_global_importance, use_container_width=True)
                else:
                    st.warning("Global feature importance data not available.")
            
            # Column 2 & 3: Prediction Distribution Pie Chart
            with col2:
                if 'prediction_distribution' in combined:
                    st.markdown("#### 🔄 Prediction Distribution")
                    labels = list(combined['prediction_distribution'].keys())
                    values = list(combined['prediction_distribution'].values())
                    fig_pie = px.pie(
                        names=labels,
                        values=values,
                        title=None,
                        color=labels,
                        color_discrete_map={'Normal': '#007BFF', 'Anomaly': '#FF4B4B'}
                    )
                    fig_pie.update_layout(
                        height=330, 
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning("Prediction distribution data not available.")

            # Column 3: Correlation Matrix
            with col3:
                if 'correlation_matrix' in combined:
                    st.markdown("#### 🔄 Correlation Matrix")
                    # Filter to important features only to make it more readable
                    important_features = ["short_call_ratio", "unique_called_ratio", "pct_daytime", "mean_duration", "pct_weekend", "unanswered_pct"]
                    filtered_corr = {k: {k2: v2 for k2, v2 in v.items() if k2 in important_features} 
                                    for k, v in combined['correlation_matrix'].items() 
                                    if k in important_features}
                    
                    corr_df = pd.DataFrame.from_dict(filtered_corr)
                    fig_corr = px.imshow(
                        corr_df,
                        color_continuous_scale='RdBu_r',
                        zmin=-1, 
                        zmax=1,
                        title=None
                    )
                    fig_corr.update_layout(
                        height=330,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.warning("Correlation matrix data not available.")

            # Second row - two columns
            col1, col2 = st.columns(2)
            
            # Column 1: Feature Distribution
            with col1:
                if 'feature_distributions' in combined:
                    st.markdown("#### 📈 Feature Distribution")
                    
                    # Create a select box but more compact
                    feature_options = list(combined['feature_distributions'].keys())
                    select_feature = st.selectbox(
                        "Select feature:", 
                        options=feature_options,
                        key="compact_feature_selector"
                    )
                    
                    if select_feature:
                        feature_dist = combined['feature_distributions'][select_feature]
                        normal_values = feature_dist['normal']
                        anomaly_values = feature_dist['anomaly']
                        
                        # Use more compact visualization
                        stats_to_show = ['mean', '25%', '50%', '75%']
                        
                        fig_dist = go.Figure()
                        fig_dist.add_trace(go.Bar(
                            x=[normal_values[s] for s in stats_to_show],
                            y=stats_to_show,
                            orientation='h',
                            name="Normal",
                            marker_color='#007BFF'
                        ))
                        fig_dist.add_trace(go.Bar(
                            x=[anomaly_values[s] for s in stats_to_show],
                            y=stats_to_show,
                            orientation='h',
                            name="Anomaly",
                            marker_color='#FF4B4B'
                        ))
                        fig_dist.update_layout(
                            title=None,
                            xaxis_title="Value",
                            barmode='group',
                            height=300,
                            margin=dict(l=10, r=10, t=30, b=30),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.warning("Feature distribution data not available.")
            
            # Column 2: Anomaly Score Distribution
            with col2:
                if 'anomaly_score_distribution' in combined:
                    st.markdown("#### 🔔 Anomaly Score Distribution")
                    hist_data = combined['anomaly_score_distribution']['histogram_data']
                    bins = hist_data['bins']
                    # Use fewer bins for cleaner visualization
                    bin_indices = range(0, len(bins)-1, 2)  # Take every other bin
                    bin_centers = [(bins[i] + bins[i+1])/2 for i in bin_indices if i+1 < len(bins)]
                    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in bin_indices if i+1 < len(bins)]
                    
                    # Aggregate counts for the reduced bins
                    normal_counts = []
                    anomaly_counts = []
                    for i in bin_indices:
                        if i+1 < len(bins):
                            if i < len(hist_data['normal_counts']):
                                normal_counts.append(hist_data['normal_counts'][i])
                            else:
                                normal_counts.append(0)
                            if i < len(hist_data['anomaly_counts']):
                                anomaly_counts.append(hist_data['anomaly_counts'][i])
                            else:
                                anomaly_counts.append(0)
                    
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Bar(
                        x=bin_centers,
                        y=normal_counts,
                        name='Normal',
                        marker_color='#007BFF',
                        text=bin_labels
                    ))
                    fig_hist.add_trace(go.Bar(
                        x=bin_centers,
                        y=anomaly_counts,
                        name='Anomaly',
                        marker_color='#FF4B4B',
                        text=bin_labels
                    ))
                    fig_hist.update_layout(
                        title=None,
                        xaxis_title="Anomaly Score",
                        yaxis_title="Count",
                        barmode='group',
                        height=300,
                        margin=dict(l=10, r=10, t=30, b=30),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.warning("Anomaly score distribution data not available.")
