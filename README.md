# 📘 Credit Risk Model API — Logistic Regression (FastAPI)

A production-style **credit risk prediction API** built using **FastAPI** and **scikit-learn**.  
This project demonstrates an end-to-end machine learning workflow including preprocessing, scaling, one-hot encoding, and real-time prediction through a REST API.

---

## 🚀 Features
- Real-time credit risk prediction  
- FastAPI backend with `/predict` endpoint  
- Logistic Regression ML model  
- Automatic preprocessing (encoding + scaling)  
- Clean project structure  
- Easy to run locally  

---

## 📁 Project Structure
```
CreditRiskModel/
│── app/
│   └── main.py
│
│── model/
│   ├── logistic_credit_model.pkl     # ignored by git
│   ├── scaler.pkl                    # ignored by git
│   └── train_columns.json
│
│── data/                             # optional
│── requirements.txt
│── README.md
│── .gitignore
│── venv/                             # ignored
```

---

## ▶️ How to Run Locally

### 1. Clone this repository
```bash
git clone https://github.com/vmlrj02/creditRiskModel-logisticRegression.git
cd creditRiskModel-logisticRegression
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add model artifacts  
Place these files inside the `model/` directory:
- logistic_credit_model.pkl  
- scaler.pkl  
- train_columns.json  

### 5. Start FastAPI server
```bash
uvicorn app.main:app --reload
```

### 6. Open API docs
👉 http://127.0.0.1:8000/docs

---

## 🧪 Example Input
```json
{
  "data": {
    "person_age": 35,
    "person_income": 55000,
    "person_emp_length": 5,
    "loan_amnt": 10000,
    "loan_int_rate": 12.5,
    "loan_percent_income": 0.18,
    "cb_person_cred_hist_length": 4,
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "loan_grade": "C",
    "cb_person_default_on_file": "N"
  }
}
```

### Example Output
```json
{
  "prediction": 0,
  "probability": 0.1035
}
```

---

## 🧠 Model Overview
The logistic regression model uses:

### Numerical Features
- person_age  
- person_income  
- person_emp_length  
- loan_amnt  
- loan_int_rate  
- loan_percent_income  
- cb_person_cred_hist_length  

### One-hot Encoded Categorical Features
- person_home_ownership  
- loan_intent  
- loan_grade  
- cb_person_default_on_file  

The API automatically:
- Encodes categories  
- Aligns columns to training schema  
- Scales features  
- Returns both prediction + probability  

---

## 📌 Purpose of This Project
This project demonstrates:
- End-to-end ML development  
- Integrating ML into a real backend API  
- Data preprocessing + scaling pipelines  
- Clean software engineering & deployment practices  


---

## 🛠️ Tech Stack
- Python 3  
- FastAPI  
- Uvicorn  
- scikit-learn  
- Pandas  
- NumPy  

---

## 📈 Optional Future Improvements
- Add SHAP explainability  
- Add unit tests  
- Add CI/CD  
- Deploy to Render / Railway  
- Add Dockerfile  
