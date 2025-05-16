import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import openai
from databricks import sql
import os

# --- CONFIG ---
st.set_page_config(page_title="📞 Telecom Fraud Detection", layout="wide")

# --- LOAD OPENAI KEY ---
openai.api_key = st.secrets["openai_api_key"]

# --- LOAD MODEL ---
iso_model = joblib.load("isolation_forest_model.joblib")

# --- INPUT PHONE NUMBER ---
phone_input = st.text_input("Enter phone number to analyze")

if phone_input:
    # --- CONNECT TO DATABRICKS ---
    connection = sql.connect(
        server_hostname=st.secrets["databricks_host"],
        http_path=st.secrets["databricks_http_path"],
        access_token=st.secrets["databricks_token"]
    )
    
    cursor = connection.cursor()
    
    # Update schema.table to match your actual DB table
    query = f"SELECT * FROM your_catalog.your_schema.telecom_data WHERE caller = '{phone_input}'"
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    df = pd.DataFrame(rows, columns=columns)
    
    if df.empty:
        st.warning("No data found for this phone number.")
    else:
        # --- MODEL PREDICTIONS ---
        feature_cols = [...]  # replace with your actual features
        sample_rows = df[feature_cols]
        predictions = iso_model.predict(sample_rows)
        mapped_preds = pd.Series(predictions, index=sample_rows.index).map({1: 'Normal', -1: 'Anomaly'})
        anomaly_scores = iso_model.decision_function(sample_rows) * -1
        
        df['prediction'] = mapped_preds
        df['anomaly_score'] = anomaly_scores
        
        # --- SHAP ---
        explainer = shap.Explainer(iso_model, sample_rows)
        shap_values = explainer(sample_rows)
        
        # --- TOP 5 Anomalies and Normals ---
        anomalies = df[df['prediction'] == 'Anomaly'].sort_values(by='anomaly_score', ascending=False).head(5)
        normals = df[df['prediction'] == 'Normal'].sort_values(by='anomaly_score', ascending=True).head(5)
        combined = pd.concat([anomalies, normals])
        
        # --- EXPLANATIONS ---
        explanations = []
        for _, row in combined.iterrows():
            prompt = (
                f"This is a telecom fraud detection system. Based on the features: "
                f"{', '.join([f'{col}={row[col]}' for col in feature_cols])}, "
                f"explain in a sentence why caller {row['caller']} is labeled '{row['prediction']}'."
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
                explanation = response['choices'][0]['message']['content'].strip()
            except Exception as e:
                explanation = f"OpenAI API error: {e}"
            explanations.append(explanation)

        combined['explanation'] = explanations

        # --- DISPLAY TABLE ---
        st.subheader("📊 Prediction Summary")
        st.dataframe(combined[['caller', 'prediction', 'explanation']])

        # --- SHAP PLOTS ---
        st.subheader("🔍 SHAP Waterfall Plots")
        shap_combined_values = explainer(combined[feature_cols])
        for i, row in enumerate(combined.itertuples(), 1):
            st.markdown(f"**Caller:** {row.caller} &nbsp;&nbsp;&nbsp; **Prediction:** {row.prediction}")
            shap.plots.waterfall(shap_combined_values[i - 1], max_display=10, show=False)
            st.pyplot(plt.gcf())
            plt.clf()
