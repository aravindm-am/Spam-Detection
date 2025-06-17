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

API_BASE = "http://163.69.82.203:8095/tmf/v1"

# Streamlit UI
# Remove blank spaces before the title by injecting CSS to set margin-top: 0 for .block-container and .main
st.markdown('''
<style>
.block-container { margin-top: 0 !important; padding-top: 0 !important; }
section.main { padding-top: 0 !important; }
header[data-testid="stHeader"] { margin-bottom: 0 !important; padding-bottom: 0 !important; }
.custom-header-box { margin-top: 14px !important; }
</style>
''', unsafe_allow_html=True)

# --- Custom header with image on the right ---
st.markdown('''
<div class="custom-header-box">
  <div class="custom-header-title">
    <span style="font-size:2.8rem;font-weight:800;color:#1a237e;vertical-align:middle;">📞 Telecom Fraud Detection</span>
  </div>
  <div class="custom-header-img">
    <img src="https://passionateaboutoss.com/directory/wp-content/uploads/2019/09/Subex_logo_png-397112561.png" alt="Telecom Logo" style="height:64px;width:auto;object-fit:contain;" />
  </div>
</div>
<style>
.custom-header-box {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #f7f9fb;
  border-radius: 24px;
  box-shadow: 0 2px 12px rgba(30, 34, 90, 0.07);
  padding: 32px 40px 24px 32px;
  margin-bottom: 18px;
  margin-top: 0;
  min-height: 80px;
}
.custom-header-title {
  flex: 1;
  display: flex;
  align-items: center;
}
.custom-header-img {
  flex-shrink: 0;
  margin-left: 32px;
  display: flex;
  align-items: center;
  height: 64px;
}
@media (max-width: 700px) {
  .custom-header-box { flex-direction: column; align-items: flex-start; padding: 18px 12px 12px 12px; }
  .custom-header-img { margin-left: 0; margin-top: 12px; }
}
</style>
''', unsafe_allow_html=True)

# --- Custom CSS for full-width layout, removing centering, and reducing top spacing ---
st.markdown('''
<style>
/* Remove Streamlit's default max-width and centering */
section.main > div { max-width: 100vw !important; padding-left: 0 !important; padding-right: 0 !important; }
.block-container { max-width: 100vw !important; padding-left: 2vw !important; padding-right: 2vw !important; }
.css-18e3th9 { align-items: stretch !important; }
.stAlert:first-child { display: none !important; }

body, .main, .block-container {
    background: #f7f9fb !important;
    font-family: 'Segoe UI', 'Roboto', Arial, sans-serif;
}

# /* Remove blank spaces before the title */
# .block-container { margin-top: 0 !important; }
# section.main { padding-top: 0 !important; }

# /* Card-like containers */
# .stContainer, .st-cb, .st-bb, .st-cg, .st-cf, .st-cd, .st-ce {
#     background: #fff !important;
#     border-radius: 18px !important;
#     box-shadow: 0 2px 12px 0 rgba(0,0,0,0.07) !important;
#     padding: 24px 24px 16px 24px !important;
#     margin-bottom: 18px !important;
# }

# /* Remove empty card/box below tabs */
# .stContainer:empty, .st-cb:empty, .st-bb:empty, .st-cg:empty, .st-cf:empty, .st-cd:empty, .st-ce:empty {
#     display: none !important;
#     height: 0 !important;
#     min-height: 0 !important;
#     margin: 0 !important;
#     padding: 0 !important;
#     border: none !important;
#     box-shadow: none !important;
# }

/* Section headers */
h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    color: #1a237e !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #007BFF 0%, #0056b3 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.6em 2em !important;
    box-shadow: 0 2px 8px 0 rgba(0,123,255,0.08) !important;
    transition: background 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #0056b3 0%, #007BFF 100%) !important;
}

/* File uploader */
.stFileUploader, .stFileUploader label {
    background: #f0f4fa !important;
    border-radius: 10px !important;
    border: 1.5px solid #e3e8f0 !important;
    padding: 1em !important;
    color: #1a237e !important;
    font-weight: 500 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1a237e !important;
    background: #e3e8f0 !important;
    border-radius: 8px 8px 0 0 !important;
    margin-right: 4px !important;
}
.stTabs [aria-selected="true"] {
    background: #fff !important;
    color: #007BFF !important;
    border-bottom: 2.5px solid #007BFF !important;
}

# /* Info/success/warning banners */
# .stAlert {
#     border-radius: 10px !important;
#     font-size: 1.05rem !important;
# }

# # /* Pie/Bar/Plotly chart container tweaks */
# # .stPlotlyChart, .stPlotlyChart > div {
# #     background: #fff !important;
# #     border-radius: 14px !important;
# #     box-shadow: 0 2px 8px 0 rgba(0,0,0,0.04) !important;
# #     padding: 8px !important;
# # }

# /* Input fields */
# .stTextInput > div > input {
#     border-radius: 8px !important;
#     border: 1.5px solid #e3e8f0 !important;
#     background: #f0f4fa !important;
#     font-size: 1.08rem !important;
#     padding: 0.5em 1em !important;
# }

/* Hide Streamlit default hamburger and footer */
#MainMenu, footer {visibility: hidden;}
</style>
''', unsafe_allow_html=True)

# # Apply custom CSS to left-align the app and optimize layout
# st.markdown("""
# <style>
#     html, body, .block-container {
#         width: 100vw !important;
#         max-width: 100vw !important;
#         min-width: 100vw !important;
#         margin: 0 !important;
#         padding: 0 !important;
#         box-sizing: border-box !important;
#     }
#     .block-container {
#         padding-top: 0.5rem !important;
#         padding-right: 1rem !important;
#         padding-left: 1rem !important;
#         padding-bottom: 0.5rem !important;
#     }
#     .stPlotlyChart {
#         margin-bottom: 0 !important;
#     }
#     /* Remove Streamlit's default centering and width restrictions */
#     [data-testid="stAppViewContainer"] > .main {
#         max-width: 100vw !important;
#         width: 100vw !important;
#         margin-left: 0 !important;
#         margin-right: 0 !important;
#         padding-left: 0 !important;
#         padding-right: 0 !important;
#     }
#     [data-testid="stHeader"] {
#         max-width: 100vw !important;
#         width: 100vw !important;
#         margin: 0 !important;
#         padding: 0 !important;
#     }
#     [data-testid="stSidebar"] {
#         margin: 0 !important;
#         padding: 0 !important;
#     }
# </style>
# """, unsafe_allow_html=True)

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
    },    "prediction_distribution": {
        "Normal": 9582,
        "Anomaly": 942
    },
    "spam_prefix_bar_plot": {
        "prefixes": [
            "80184",
            "76717",
            "92044",
            "88311",
            "72679"
        ],
        "counts": [
            10,
            7,
            5,
            12,
            4
        ]
    },
    "time_call": {
        "Weekday": 6559,
        "Weekend": 3441
    }
}

# Load Databricks secrets
DATABRICKS_HOST = st.secrets["databricks_host"]
DATABRICKS_PATH = st.secrets["databricks_http_path"]
DATABRICKS_TOKEN = st.secrets["databricks_token"]
DATABRICKS_NOTEBOOK_PATH = st.secrets["databricks_notebook_path"]
DATABRICKS_NOTEBOOK_PATH_BATCH = st.secrets["databricks_notebook_path_batch"]

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
            "notebook_path": DATABRICKS_NOTEBOOK_PATH,  # Use individual notebook path from secrets
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

# # Streamlit UI
# # Remove blank spaces before the title by injecting CSS to set margin-top: 0 for .block-container and .main
# st.markdown('''
# <style>
# .block-container { margin-top: 0 !important; }
# section.main { padding-top: 0 !important; }
# </style>
# ''', unsafe_allow_html=True)
# st.title("📞 Telecom Fraud Detection")

# Change the order of tabs - Combined Analysis first, Individual Analysis second
api_tabs = st.tabs(["📊 Combined Analysis", "🔎 Individual Analysis", "🔗 Blockchain"])

# --- Responsive height: inject JS to get viewport height and set in session_state ---
if 'viewport_height' not in st.session_state:
    st.session_state['viewport_height'] = 800  # fallback default

st.markdown('''
<script>
(function() {
    function sendHeight() {
        const height = window.innerHeight || document.documentElement.clientHeight || document.body.clientHeight;
        window.parent.postMessage({streamlitSetFrameHeight: height}, '*');
        const streamlitDoc = window.parent.document;
        if (streamlitDoc) {
            const input = streamlitDoc.getElementById('streamlit-viewport-height');
            if (input) input.value = height;
        }
    }
    window.addEventListener('resize', sendHeight);
    sendHeight();
})();
</script>
<input type="hidden" id="streamlit-viewport-height" value="800" />
''', unsafe_allow_html=True)

viewport_height = st.query_params.get('viewport_height', [None])[0]
if viewport_height:
    try:
        st.session_state['viewport_height'] = int(viewport_height)
    except:
        pass

# --- Calculate plot heights ---
# Reserve some space for header, tabs, and info boxes
header_height = 180  # px (title, tabs, info)
avail_height = st.session_state['viewport_height'] - header_height
if avail_height < 400:
    avail_height = 600  # fallback

# 2 rows: row1 (top, 1/2), row2 (bottom, 1/2)
row_height = int(avail_height / 2)
# row1: left (feature importance, 2/3), right (prediction + corr, 1/3)
row1_left = int(row_height * 0.98)
row1_right = int(row_height * 0.48)
# row2: left (feature dist), right (anomaly score)
row2_height = int(row_height * 0.98)

# Inject JavaScript to get viewport size and set in session state
viewport_js = """
<script>
(function() {
    function sendViewportSize() {
        const height = window.innerHeight || document.documentElement.clientHeight;
        const width = window.innerWidth || document.documentElement.clientWidth;
        const streamlitDoc = window.parent.document;
        streamlitDoc.dispatchEvent(new CustomEvent("streamlit:setComponentValue", {
            detail: {key: "viewport_height", value: height}
        }));
        streamlitDoc.dispatchEvent(new CustomEvent("streamlit:setComponentValue", {
            detail: {key: "viewport_width", value: width}
        }));
    }
    window.addEventListener('resize', sendViewportSize);
    sendViewportSize();
})();
</script>
"""
st.markdown(viewport_js, unsafe_allow_html=True)

# Helper to get viewport size from Streamlit session state
if 'viewport_height' not in st.session_state:
    st.session_state['viewport_height'] = 900  # fallback default
if 'viewport_width' not in st.session_state:
    st.session_state['viewport_width'] = 1600  # fallback default

# Listen for JS events to update session state
viewport_height = st.query_params.get('viewport_height', [st.session_state['viewport_height']])[0]
viewport_width = st.query_params.get('viewport_width', [st.session_state['viewport_width']])[0]
try:
    st.session_state['viewport_height'] = int(viewport_height)
    st.session_state['viewport_width'] = int(viewport_width)
except Exception:
    pass

# Tab 1: Combined Analysis (now first)
with api_tabs[0]:
    # --- Batch screening UI ---
    if st.button("Start Screening", key="start_screening_button"):
            # Just run the Databricks notebook (databricks-new.py) and display the JSON output
            headers = {
                "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                "Content-Type": "application/json"
            }
            EXISTING_CLUSTER_ID = "0521-131856-gsh3b6se"
            batch_notebook_path = DATABRICKS_NOTEBOOK_PATH_BATCH  # Path to databricks-new.py
            submit_payload = {
                "run_name": f"BatchFraudCheck_{int(time.time())}",
                "notebook_task": {
                    "notebook_path": batch_notebook_path,
                    "base_parameters": {}  # No need to pass input_file if hardcoded
                },
                "existing_cluster_id": EXISTING_CLUSTER_ID        
            }
            response = requests.post(
                f"{DATABRICKS_HOST}/api/2.1/jobs/runs/submit",
                headers=headers,
                json=submit_payload
            )
            if response.status_code != 200:
                st.error("❌ Failed to start Databricks batch job.")
                st.text(response.text)
            else:
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
                if notebook_output and "results" in notebook_output:
                  # Display results table with caller, prediction, anomaly_score, country, and operator
                    st.markdown("#### <span style='color:#007BFF;'>Scoring Results</span>", unsafe_allow_html=True)
                    results_df = pd.DataFrame(notebook_output["results"])
                    # Select and rename columns for display
                    results_df = results_df[['caller', 'prediction', 'anomaly_score', 'caller_country', 'caller_operator']]
                    results_df.columns = ['Caller', 'Prediction', 'Anomaly Score', 'Country', 'Operator']
                    # Format anomaly scores to two decimal places as string
                    results_df['Anomaly Score'] = results_df['Anomaly Score'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "")
                    # Sort: Anomaly first, then Normal
                    anomaly_rows = results_df[results_df['Prediction'] == 'Anomaly']
                    normal_rows = results_df[results_df['Prediction'] == 'Normal']
                    results_df = pd.concat([anomaly_rows, normal_rows], ignore_index=True)

                    # Compact table CSS
                    st.markdown("""
                        <style>
                        .compact-table td, .compact-table th {
                            padding: 0.25rem 0.5rem !important;
                            font-size: 0.95rem !important;
                            text-align: left !important;
                        }
                        .compact-table th { background: #f0f4fa; }
                        </style>
                    """, unsafe_allow_html=True)

                    # Render compact HTML table with new columns
                    html = '<table class="compact-table" style="width:100%;border-collapse:collapse;">'
                    html += '<tr><th>Caller</th><th>Prediction</th><th>Anomaly Score</th><th>Country</th><th>Operator</th></tr>'
                    for _, row in results_df.iterrows():
                        color = "#FF4B4B" if row["Prediction"] == 'Anomaly' else "#1a237e"
                        html += f'<tr>' \
                                f'<td style="color:{color};">{row["Caller"]}</td>' \
                                f'<td style="color:{color};">{row["Prediction"]}</td>' \
                                f'<td style="color:{color};">{row["Anomaly Score"]}</td>' \
                                f'<td style="color:{color};">{row["Country"]}</td>' \
                                f'<td style="color:{color};">{row["Operator"]}</td>' \
                                f'</tr>'
                    html += '</table>'
                    st.markdown(html, unsafe_allow_html=True)

                    # Save the raw results for use in Blockchain tab
                    st.session_state['scoring_results'] = notebook_output["results"]

                    # --- Populate anomaly_numbers in session_state for Blockchain tab ---
                    anomaly_dict = {}
                    for _, row in results_df.iterrows():
                        if row['Prediction'] == 'Anomaly':
                            try:
                                score = float(row['Anomaly Score'])
                            except:
                                score = 0.0
                            anomaly_dict[row['Caller']] = score
                    st.session_state['anomaly_numbers'] = anomaly_dict

                    # # --- Add to blockchain feature: just redirect to Blockchain tab ---
                    # st.markdown("<br>", unsafe_allow_html=True)
                    # if st.button("Add to blockchain", key="add_to_blockchain_btn"):
                    #     st.session_state['switch_to_blockchain_tab'] = True
                    #     st.experimental_rerun()

                    # --- Show preview of sample_predictions.csv from GitHub ---
                    try:
                        csv_url = "https://raw.githubusercontent.com/aravindm-am/Spam-Detection-Test/main/spam_call_dataset_updated.csv"  # Update with your actual username/repo if needed
                        sample_df = pd.read_csv(csv_url)
                        st.markdown("#### <span style='color:#007BFF;'>CSV Preview</span>", unsafe_allow_html=True)
                        st.dataframe(sample_df.head(10), use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not load sample_predictions.csv: {e}")
                else:
                    st.warning("No results found in notebook output.")
                    
    # Always show the hardcoded plots below the upload UI
    # Check if we have a real analysis or should use the hardcoded data
    if 'shap_data' in st.session_state and 'combined_analysis' in st.session_state.shap_data:
        shap_data = st.session_state.shap_data
        combined = shap_data['combined_analysis']
        st.success("✅ Displaying analysis from the latest run")
    else:
        # Use hardcoded combined analysis
        combined = HARDCODED_COMBINED_ANALYSIS
        st.info("ℹ️ Displaying pre-computed analysis. Run an individual analysis for real-time data.")

    # Main container for the combined analysis layout
    with st.container():
        # Calculate available height for plots
        header_height = 60  # app title + info banner
        padding = 32  # extra margin/padding
        available_height = st.session_state['viewport_height'] - header_height - padding
        # 2 rows: each row gets half the available height
        row_height = max(200, int(available_height / 2))
        # 3 columns for each row
        col_width = int(st.session_state['viewport_width'] / 3)

        # --- ROW 1 ---
        row1_col1, row1_col2, row1_col3 = st.columns(3, gap="medium")
        with row1_col1:
            if 'global_feature_importance' in combined:
                st.markdown("#### <span style='color:#007BFF;'>📊 Top Indicators of Fraudulent Activity</span>", unsafe_allow_html=True)
                global_importance_df = pd.DataFrame({
                    'Feature': list(combined['global_feature_importance'].keys()),
                    'Importance': list(combined['global_feature_importance'].values())
                }).sort_values('Importance', ascending=False)
                global_importance_df = global_importance_df.head(10)
                fig_global_importance = px.bar(
                    global_importance_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig_global_importance.update_layout(
                    height=row_height,
                    margin=dict(l=10, r=10, t=10, b=10),
                    title="",
                    font=dict(size=14, family='Segoe UI', color='#1a237e'),
                    plot_bgcolor='white',
                )
                st.plotly_chart(fig_global_importance, use_container_width=True)
            else:
                st.warning("Global feature importance data not available.")
        with row1_col2:
            if 'prediction_distribution' in combined:
                st.markdown("#### <span style='color:#007BFF;'>🔄 Fraud vs. Normal Call Distribution</span>", unsafe_allow_html=True)
                labels = list(combined['prediction_distribution'].keys())
                values = list(combined['prediction_distribution'].values())
                fig_pie = px.pie(
                    names=labels,
                    values=values,
                    color=labels,
                    color_discrete_map={'Normal': '#007BFF', 'Anomaly': '#FF4B4B'},
                    hole=0.4
                )
                fig_pie.update_layout(
                    height=row_height,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                    title="",
                    font=dict(size=14, family='Segoe UI', color='#1a237e'),
                    plot_bgcolor='white',
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.warning("Prediction distribution data not available.")
        with row1_col3:
            if 'spam_prefix_bar_plot' in combined:
                st.markdown("#### <span style='color:#007BFF;'>📞 Spam Call Frequency by Number Prefix</span>", unsafe_allow_html=True)
                prefix_data = combined['spam_prefix_bar_plot']
                fig_prefix = go.Figure(go.Bar(
                    x=prefix_data['prefixes'],
                    y=prefix_data['counts'],
                    marker=dict(
                        color='#FF7F50',  # Coral color
                        line=dict(width=1, color='#FF6347')  # Tomato color border
                    ),
                    hovertemplate='Prefix: %{x}<br>Count: %{y}<extra></extra>'
                ))
                fig_prefix.update_layout(
                    xaxis_title=dict(
                        text="Number Prefix",
                        font=dict(size=14, color='#1a237e')
                    ),
                    yaxis_title=dict(
                        text="Number of Anomalous Callers",
                        font=dict(size=14, color='#1a237e')
                    ),
                    height=row_height,
                    margin=dict(l=20, r=20, t=40, b=60),  # Increased bottom margin for labels
                    bargap=0.2,  # Add some gap between bars
                    yaxis=dict(
                        gridcolor='rgba(0,0,0,0.1)',
                        gridwidth=1,
                        griddash='dash',
                        tickformat='',  # Display full numbers without 'k' suffix
                        tickfont=dict(size=12)
                    ),
                    xaxis=dict(
                        tickangle=0,  # Horizontal labels
                        tickmode='array',
                        ticktext=prefix_data['prefixes'],
                        tickvals=prefix_data['prefixes'],
                        tickfont=dict(size=12),
                        showgrid=False
                    ),
                    plot_bgcolor='white',  # White background
                    showlegend=False,
                    font=dict(size=14, family='Segoe UI', color='#1a237e'),
                )
                st.plotly_chart(fig_prefix, use_container_width=True)
            else:
                st.warning("Spam prefix data not available.")

        # --- ROW 2 ---
        row2_col1, row2_col2, row2_col3 = st.columns(3, gap="medium")
        with row2_col1:
            if 'correlation_matrix' in combined:
                st.markdown("#### <span style='color:#007BFF;'>🔄 Correlated Call Patterns</span>", unsafe_allow_html=True)
                important_features = ["short_call_ratio", "mean_duration", "pct_daytime", "pct_weekend"]
                filtered_corr = {k: {k2: v2 for k2, v2 in v.items() if k2 in important_features} 
                                for k, v in combined['correlation_matrix'].items() 
                                if k in important_features}
                corr_df = pd.DataFrame.from_dict(filtered_corr)
                fig_corr = px.imshow(
                    corr_df,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, 
                    zmax=1,
                    text_auto='.2f'
                )
                fig_corr.update_layout(
                    height=row_height,
                    margin=dict(l=10, r=10, t=10, b=10),
                    title="",
                    font=dict(size=14, family='Segoe UI', color='#1a237e'),
                    plot_bgcolor='white',
                )
                fig_corr.update_traces(texttemplate="%{text}", textfont={"size": 12})
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("Correlation matrix data not available.")
        with row2_col2:
            if 'feature_distributions' in combined:
                st.markdown("#### <span style='color:#007BFF;'>📈 Spotting Risk Through Call Behavior</span>", unsafe_allow_html=True)
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
                        title="",
                        xaxis_title="Value",
                        barmode='group',
                        height=row_height,
                        margin=dict(l=10, r=10, t=10, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        font=dict(size=14, family='Segoe UI', color='#1a237e'),
                        plot_bgcolor='white',
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
            else:
                st.warning("Feature distribution data not available.")
        with row2_col3:
            if 'anomaly_score_distribution' in combined:
                st.markdown("#### <span style='color:#007BFF;'>🔔 Likelihood of Fraud Across Users</span>", unsafe_allow_html=True)
                hist_data = combined['anomaly_score_distribution']['histogram_data']
                bins = hist_data['bins']
                bin_indices = range(0, len(bins)-1, 2)
                bin_centers = [(bins[i] + bins[i+1])/2 for i in bin_indices if i+1 < len(bins)]
                bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in bin_indices if i+1 < len(bins)]
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
                    title="",
                    xaxis_title="Anomaly Score",
                    yaxis_title="Count",
                    barmode='group',
                    height=row_height,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    font=dict(size=14, family='Segoe UI', color='#1a237e'),
                    plot_bgcolor='white',
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("Anomaly score distribution data not available.")

        # --- Time of Call Pie Chart ---
        if 'time_call' in combined:
            st.markdown("#### <span style='color:#007BFF;'>🕒 Time of Call Distribution</span>", unsafe_allow_html=True)
            time_labels = list(combined['time_call'].keys())
            time_values = list(combined['time_call'].values())
            fig_time_call = px.pie(
                names=time_labels,
                values=time_values,
                color=time_labels,
                color_discrete_map={'Weekday': '#007BFF', 'Weekend': '#FF4B4B'},
                hole=0.4
            )
            fig_time_call.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                title="",
                font=dict(size=14, family='Segoe UI', color='#1a237e'),
                plot_bgcolor='white',
            )
            st.plotly_chart(fig_time_call, use_container_width=True)
# Tab 2: Individual Analysis (now second)
with api_tabs[1]:
    st.markdown("#### <span style='color:#007BFF;'>Check a Phone Number for Fraud</span>", unsafe_allow_html=True)
    phone_number = st.text_input("Enter Phone Number to Check")
    run_button = st.button("Run Fraud Check", key="run_check_button")
    
    if run_button:
        st.session_state['shap_data'] = None  # or {}
        if phone_number.strip():
            with st.spinner("Subex Spam Scoring Started..."):
                result, notebook_output = run_notebook(phone_number.strip())
                if result == "SUCCESS":
                    st.success("🎉 Analysis complete!")
                    # After getting notebook_output (the JSON string)
                    shap_data = notebook_output
                    st.session_state['shap_data'] = shap_data

                    st.subheader("📞 Prediction Summary")
                    st.markdown(f"<span style='font-size:1.1rem;color:#374151;'><b>Phone Number</b>: <code>{phone_number}</code></span>", unsafe_allow_html=True)
                    not_found = False
                    if 'prediction' in shap_data and shap_data['prediction'] is not None:
                        st.markdown(f"<span style='font-size:1.1rem;color:#374151;'><b>Prediction</b>: <code>{shap_data['prediction']}</code></span>", unsafe_allow_html=True)
                    else:
                        st.warning("Prediction not available for this number.")
                        not_found = True
                    if 'anomaly_score' in shap_data and shap_data['anomaly_score'] is not None:
                        st.markdown(f"<span style='font-size:1.1rem;color:#374151;'><b>Anomaly Score</b>: <code>{shap_data['anomaly_score']:.4f}</code></span>", unsafe_allow_html=True)
                    if 'explanation' in shap_data and shap_data['explanation']:
                        st.markdown(f"<span style='font-size:1.1rem;color:#374151;'><b>AI Explanation</b>: {shap_data['explanation']}</span>", unsafe_allow_html=True)

                    # Only show feature importance if present in shap_data
                    if 'feature_importance' in shap_data and shap_data['feature_importance']:
                        feature_importance_df = pd.DataFrame({
                            'Feature': list(shap_data['feature_importance'].keys()),
                            'Importance': list(shap_data['feature_importance'].values())
                        }).sort_values('Importance', ascending=False)

                        # Prepare data for waterfall plot
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
                                    color='Importance',
                                    color_continuous_scale='Blues'
                                )
                            fig_importance.update_layout(title="Individual Feature Importance")
                            st.plotly_chart(fig_importance, use_container_width=True)
                                              

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
                            st.plotly_chart(fig_waterfall, use_container_width=True)
                    else:
                        not_found = True
                    if not_found:
                        st.info("Number not found in dataset.")
        else:
            st.warning("📱 Please enter a valid phone number.")

# Tab 3: Blockchain API Interface
with api_tabs[2]:
    st.title("QoT Record Interface")
    mode = st.selectbox("Select Operation", ["Insert/Update", "Read/Query"])
    # Switch to Blockchain tab if requested
    if st.session_state.get('switch_to_blockchain_tab', False):
        st.session_state['switch_to_blockchain_tab'] = False
        st.session_state['blockchain_tab_selected'] = True
        st.experimental_set_query_params(tab=2)
    # Pre-select anomaly if coming from Combined Analysis
    anomaly_numbers = st.session_state.get('anomaly_numbers', {})
    selected_anomaly = None
    if 'selected_anomaly_for_blockchain' in st.session_state:
        selected_anomaly = st.session_state.pop('selected_anomaly_for_blockchain')
    anomaly_score = anomaly_numbers[selected_anomaly] if selected_anomaly and selected_anomaly in anomaly_numbers else 0.1432
    # --- Store submitted MSISDNs in session state ---
    if 'submitted_msisdns' not in st.session_state:
        st.session_state['submitted_msisdns'] = []
    if mode == "Insert/Update" and anomaly_numbers:
        st.markdown("**Select an anomaly number:**")
        selected_anomaly = st.selectbox("Anomaly Numbers", list(anomaly_numbers.keys()), key="anomaly_select")
        msisdn = str(selected_anomaly) if selected_anomaly else ""
        anomaly_score = anomaly_numbers[selected_anomaly] if selected_anomaly else 0.1432
        # --- Auto-populate Operator and Country from scoring results ---
        # Try to get the last scoring results from session_state
        operator = ""
        country = ""
        if 'scoring_results' in st.session_state:
            # scoring_results is a list of dicts
            for r in st.session_state['scoring_results']:
                if str(r.get('caller')) == msisdn:
                    operator = r.get('caller_operator', "")
                    country = r.get('caller_country', "")
                    break
        else:
            # fallback: try to get from anomaly_dict if available
            operator = ""
            country = ""
    else:
        msisdn = ""
        operator = ""
        country = ""
    if mode == "Insert/Update":
        st.subheader("Insert or Update QoT Record")
        src_o = st.text_input("Source Operator", operator or "Jio")
        src_c = st.text_input("Source Country", country or "India")
        rep_o = "Airtel"   #st.text_input("Reported Operator", "Airtel")
        rep_c = "India     "#st.text_input("Reported Country", "India")
        score = st.number_input("Score", min_value=0.0, max_value=1.0, value=anomaly_score, step=0.01, key="score_input")
        if st.button("Submit"):
            payload = {
                "requestId": "000001",
                "module": "tmforum",
                "channelID": "globalspamdatachannel",
                "chaincodeID": "qotcc",
                "functionName": "addQoTRecord",
                "payload": {
                    "msisdn": str(msisdn),
                    "src_o": src_o,
                    "src_c": src_c,
                    "rep_o": rep_o,
                    "rep_c": rep_c,
                    "score": score
                }
            }
            try:
                response = requests.post(f"{API_BASE}/invoke/", headers={"Content-Type": "application/json"}, data=json.dumps(payload))
                st.code(response.text, language="json")
                if msisdn and msisdn not in st.session_state['submitted_msisdns']:
                    st.session_state['submitted_msisdns'].append(msisdn)
            except Exception as e:
                st.error(f"Error: {e}")
    elif mode == "Read/Query":
        st.subheader("Read QoT Record")
        msisdn_options = st.session_state['submitted_msisdns']
        msisdn_selected = None
        if msisdn_options:
            msisdn_selected = st.selectbox("Select previously submitted MSISDN", options=msisdn_options, key="read_msisdn_select")
        else:
            st.info("No MSISDNs have been submitted yet.")
        msisdn_to_query = str(msisdn_selected) if msisdn_selected else ""
        record_response = None
        if st.button("Fetch Record") and msisdn_to_query:
            payload = {
                "requestId": "000001",
                "module": "tmforum",
                "channelID": "globalspamdatachannel",
                "chaincodeID": "qotcc",
                "functionName": "getQoTRecord",
                "payload": [str(msisdn_to_query)]
            }
            try:
                response = requests.post(f"{API_BASE}/query/", headers={"Content-Type": "application/json"}, data=json.dumps(payload))
                record_response = response.text
                st.code(record_response, language="json")
            except Exception as e:
                st.error(f"Error: {e}")
        # if 'record_response' in locals() and record_response:
        #     st.markdown("#### QoT Record Result")
        #     st.code(record_response, language="json")
