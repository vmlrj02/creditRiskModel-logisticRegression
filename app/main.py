# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

# ==============================
# Resolve paths (robust)
# ==============================
# Project root = two levels up from this file (CreditRiskModel/)
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH  = BASE_DIR / "model" / "logistic_credit_model.joblib"
SCALER_PATH = BASE_DIR / "model" / "scaler.joblib"
COLUMNS_PATH = BASE_DIR / "model" / "train_columns.json"

# Helpful debug if things fail when starting via uvicorn
# (you can comment out the print statements later)
print("DEBUG: BASE_DIR =", BASE_DIR)
print("DEBUG: MODEL_PATH =", MODEL_PATH)
print("DEBUG: SCALER_PATH =", SCALER_PATH)
print("DEBUG: COLUMNS_PATH =", COLUMNS_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

if not SCALER_PATH.exists():
    raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

if not COLUMNS_PATH.exists():
    raise FileNotFoundError(f"Missing {COLUMNS_PATH}. You need to save training column names (see instructions).")

# load files
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(COLUMNS_PATH, "r") as f:
    train_columns = json.load(f)

# ==============================
# FastAPI setup
# ==============================
app = FastAPI(title="Credit Risk Model API")

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
