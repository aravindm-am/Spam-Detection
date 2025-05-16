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


st.header("🔌 Databricks Connection Test")

conn = get_databricks_connection()
if conn:
    try:
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        st.success("✅ Connected successfully!")
        st.write("Available Databases:", databases)
    except Exception as e:
        st.error(f"Query failed: {e}")
else:
    st.stop()

# Input from user
phone_input = st.text_input("Enter phone number to check:")
