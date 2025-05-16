import streamlit as st
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt
import openai

openai.api_key = st.secrets["openai_api_key"]

# Load model (cached)
@st.cache_resource
def load_model():
    return joblib.load("isolation_forest_model.joblib")

model = load_model()

# Load dataset locally (cached)
@st.cache_data
def load_data():
    # Replace 'caller_df.csv' with your actual data file path
    df = pd.read_csv("caller_df.csv")
    return df

feature_cols = [
    "call_duration", "num_calls", "avg_call_interval", "std_call_interval",
    "call_to_unique_numbers", "international_calls", "night_calls", "day_calls"
]

st.title("📞 Telecom Fraud Detection")

phone_input = st.text_input("Enter a phone number (caller ID):")

if phone_input:
    with st.spinner("Running prediction..."):
        caller_df = load_data()

        if 'caller' not in caller_df.columns:
            st.error("The dataset must have a 'caller' column.")
        else:
            if phone_input not in caller_df['caller'].values:
                st.warning("Phone number not found in dataset.")
            else:
                sample_rows = caller_df[feature_cols]
                predictions = model.predict(sample_rows)
                anomaly_scores = -model.decision_function(sample_rows)

                mapped_preds = pd.Series(predictions, index=sample_rows.index).map({1: 'Normal', -1: 'Anomaly'})

                result = caller_df.copy()
                result['prediction'] = mapped_preds
                result['anomaly_score'] = anomaly_scores

                row = result[result['caller'] == phone_input].iloc[0]

                explainer = shap.Explainer(model, sample_rows)
                shap_values = explainer(sample_rows)
                shap_value = shap_values[result[result['caller'] == phone_input].index[0]]

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

                output_df = pd.DataFrame([row])
                output_df['explanation'] = explanation
                output_df.to_csv('sample_predictions.csv', index=False)

                st.subheader(f"Prediction for {row['caller']}")
                st.write(f"**Prediction:** {row['prediction']}")
                st.write(f"**Anomaly Score:** {row['anomaly_score']:.4f}")
                st.write(f"**Explanation:** {explanation}")

                st.subheader("SHAP Feature Contribution")
                fig, ax = plt.subplots()
                shap.plots.waterfall(shap_value, max_display=10, show=False)
                st.pyplot(fig)
