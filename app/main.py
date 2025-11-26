# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import boto3

# ==============================
# Paths
# ==============================
BASE_DIR = Path(__file__).resolve().parent.parent

# Local cache directory for S3 downloads
LOCAL_MODEL_DIR = BASE_DIR / "model_cache"
LOCAL_MODEL_DIR.mkdir(exist_ok=True)

LOCAL_MODEL_PATH   = LOCAL_MODEL_DIR / "logistic_credit_model.joblib"
LOCAL_SCALER_PATH  = LOCAL_MODEL_DIR / "scaler.joblib"
LOCAL_COLUMNS_PATH = LOCAL_MODEL_DIR / "train_columns.json"

# ==============================
# S3 CONFIG  👉 EDIT THESE
# ==============================
S3_BUCKET_NAME = "credit-risk-model-vimal-2025"  

S3_MODEL_KEY   = "model/logistic_credit_model.joblib"
S3_SCALER_KEY  = "model/scaler.joblib"
S3_COLUMNS_KEY = "model/train_columns.json"

# ==============================
# Download from S3 if not present
# ==============================
def download_from_s3_if_not_exists():
    s3 = boto3.client("s3")

    if not LOCAL_MODEL_PATH.exists():
        print("Downloading model from S3...")
        s3.download_file(S3_BUCKET_NAME, S3_MODEL_KEY, str(LOCAL_MODEL_PATH))

    if not LOCAL_SCALER_PATH.exists():
        print("Downloading scaler from S3...")
        s3.download_file(S3_BUCKET_NAME, S3_SCALER_KEY, str(LOCAL_SCALER_PATH))

    if not LOCAL_COLUMNS_PATH.exists():
        print("Downloading columns from S3...")
        s3.download_file(S3_BUCKET_NAME, S3_COLUMNS_KEY, str(LOCAL_COLUMNS_PATH))

# ==============================
# Load model at startup
# ==============================
download_from_s3_if_not_exists()

print("DEBUG: LOCAL_MODEL_PATH =", LOCAL_MODEL_PATH)
print("DEBUG: LOCAL_SCALER_PATH =", LOCAL_SCALER_PATH)
print("DEBUG: LOCAL_COLUMNS_PATH =", LOCAL_COLUMNS_PATH)

model = joblib.load(LOCAL_MODEL_PATH)
scaler = joblib.load(LOCAL_SCALER_PATH)

with open(LOCAL_COLUMNS_PATH, "r") as f:
    train_columns = json.load(f)

# ==============================
# FastAPI setup
# ==============================
app = FastAPI(title="Credit Risk Model API (S3-backed)")

class CreditData(BaseModel):
    data: dict  # expects a dict with column_name: value

@app.post("/predict")
def predict(input_data: CreditData):
    df = pd.DataFrame([input_data.data])
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.reindex(columns=train_columns, fill_value=0)
    X_scaled = scaler.transform(df_encoded)
    prediction = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0, 1]

    return {
        "prediction": int(prediction),
        "probability": float(round(proba, 4))
    }
