import joblib
import pandas as pd

model = joblib.load("models/best_loan_model.joblib")


def predict_loan(data: dict):
    df = pd.DataFrame([data])

    pred = model.predict(df)[0]

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])

    return {
        "prediction": bool(pred),
        "decision": "APPROVED" if pred else "REJECTED",
        "probability": probability,
    }