import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
from databricks import sql

# Load Databricks secrets
DATABRICKS_HOST = st.secrets["databricks_host"]
DATABRICKS_PATH = st.secrets["databricks_http_path"]
DATABRICKS_TOKEN = st.secrets["databricks_token"]

# Configure page
st.set_page_config(page_title="Telecom Fraud Detection", layout="wide")
st.title("📞 Telecom Fraud Detection")

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
        st.error(f"Connection failed: {e}")
        return None

@st.cache_data
def load_caller_df():
    conn = get_connection()
    query = """
    CREATE TABLE IF NOT EXISTS hive_metastore.default.caller_df AS
    SELECT * FROM (SELECT * FROM your_source_table LIMIT 10000) -- Modify this line to use your actual source
    """
    with conn.cursor() as cursor:
        cursor.execute(query)

    query = "SELECT * FROM hive_metastore.default.caller_df"
    df = pd.read_sql(query, conn)
    return df

@st.cache_resource
def load_model():
    conn = get_connection()
    # Save model from DBFS or mount to local disk if not present
    model_path = "/tmp/isolation_forest_model.joblib"
    if not os.path.exists(model_path):
        query = "SELECT model_bytes FROM hive_metastore.default.models WHERE model_name = 'isolation_forest' LIMIT 1"
        with conn.cursor() as cursor:
            cursor.execute(query)
            model_blob = cursor.fetchone()
            if model_blob:
                with open(model_path, "wb") as f:
                    f.write(model_blob[0])
            else:
                st.error("Model not found in Databricks table.")
                st.stop()
    return joblib.load(model_path)

# Input from user
phone_input = st.text_input("Enter phone number to check:")

if phone_input:
    with st.spinner("Running prediction..."):
        caller_df = load_caller_df()

        if 'caller' not in caller_df.columns:
            st.error("The dataset must have a 'caller' column.")
            st.stop()

        model = load_model()

        feature_cols = [col for col in caller_df.columns if col != 'caller']
        X = caller_df[feature_cols]

        predictions = model.predict(X)
        anomaly_scores = model.decision_function(X) * -1

        caller_df['prediction'] = pd.Series(predictions, index=X.index).map({1: 'Normal', -1: 'Anomaly'})
        caller_df['anomaly_score'] = anomaly_scores

        selected = caller_df[caller_df['caller'] == phone_input]

        if selected.empty:
            st.warning("Phone number not found in the dataset.")
        else:
            selected.to_csv("sample_predictions.csv", index=False)
            st.subheader("🔍 Prediction Result")
            st.write(selected[['caller', 'prediction', 'anomaly_score'] + feature_cols])

            explainer = shap.Explainer(model, X)
            shap_values = explainer(X)
            idx = selected.index[0]
            st.subheader("🔎 SHAP Waterfall Plot")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
            st.pyplot(fig)
