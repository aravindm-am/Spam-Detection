import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import openai
import io
from databricks import sql

# --- CONFIG ---
st.set_page_config(page_title="📞 Telecom Fraud Detection", layout="wide")
openai.api_key = st.secrets["openai_api_key"]

# Load model and features
iso_model = joblib.load("isolation_forest_model.joblib")
feature_cols = ["total_calls", "short_calls", "international_calls", "short_call_pct",
                "avg_call_duration", "day_calls", "night_calls", "weekend_calls",
                "call_variance", "short_call_ratio"]

# Connect to Databricks
DATABRICKS_HOST = st.secrets["databricks_host"]
DATABRICKS_TOKEN = st.secrets["databricks_token"]
HTTP_PATH = st.secrets["http_path"]

# --- UI ---
st.title("📞 Telecom Fraud Detection App")
phone_input = st.text_input("Enter a phone number:")

if phone_input:
    with st.spinner("Fetching data from Databricks..."):
        connection = sql.connect(server_hostname=DATABRICKS_HOST,
                                 access_token=DATABRICKS_TOKEN,
                                 http_path=HTTP_PATH)
        cursor = connection.cursor()

        query = f"SELECT * FROM telecom_data WHERE caller = '{phone_input}'"
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)

    if df.empty:
        st.error("No data found for this phone number.")
    else:
        features = df[feature_cols]
        prediction = iso_model.predict(features)[0]
        score = -iso_model.decision_function(features)[0]
        label = "Anomaly" if prediction == -1 else "Normal"

        st.subheader(f"Prediction: {label}")
        st.write(f"Anomaly Score: {score:.4f}")

        with st.spinner("Generating SHAP explanation..."):
            explainer = shap.Explainer(iso_model, features)
            shap_values = explainer(features)

            st.subheader("🔍 SHAP Waterfall Plot")
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)

        with st.spinner("Asking OpenAI for explanation..."):
            prompt = (
                f"This is a telecom fraud detection system. Based on the features: "
                f"{', '.join([f'{col}={features.iloc[0][col]}' for col in feature_cols])}, "
                f"explain in a sentence why caller {phone_input} is labeled '{label}'."
            )
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a telecom fraud expert and machine learning explainer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=100
                )
                explanation = response["choices"][0]["message"]["content"].strip()
            except Exception as e:
                explanation = f"OpenAI error: {e}"

            st.subheader("💬 OpenAI Explanation")
            st.markdown(explanation)
