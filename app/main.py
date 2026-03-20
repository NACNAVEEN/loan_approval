# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from app.schemas import LoanPredictionRequest
# from app.predictor import predict_loan

# app = FastAPI(title="Loan Prediction API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.get("/")
# def home():
#     return {"message": "Loan Prediction API running"}


# @app.post("/predict")
# def predict(payload: LoanPredictionRequest):
#     result = predict_loan(payload.model_dump())
#     return result

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.schemas import LoanPredictionRequest
from app.predictor import predict_loan

app = FastAPI(title="Loan Prediction API")

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.post("/predict")
def predict(payload: LoanPredictionRequest):
    result = predict_loan(payload.model_dump())
    return result