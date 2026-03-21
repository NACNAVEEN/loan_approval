# # from fastapi import FastAPI
# # from fastapi.middleware.cors import CORSMiddleware
# # from app.schemas import LoanPredictionRequest
# # from app.predictor import predict_loan

# # app = FastAPI(title="Loan Prediction API")

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_credentials=True,
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )


# # @app.get("/")
# # def home():
# #     return {"message": "Loan Prediction API running"}


# # @app.post("/predict")
# # def predict(payload: LoanPredictionRequest):
# #     result = predict_loan(payload.model_dump())
# #     return result

# from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from app.schemas import LoanPredictionRequest
# from app.predictor import predict_loan

# app = FastAPI(title="Loan Prediction API")

# app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


# @app.get("/")
# def home():
#     return FileResponse("frontend/index.html")


# @app.post("/predict")
# def predict(payload: LoanPredictionRequest):
#     result = predict_loan(payload.model_dump())
#     return result

from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.schemas import (
    LoanPredictionRequest,
    LoanPredictionResponse,
    LoanPredictionHistory
)
from app.predictor import predict_loan
from app.database import Base, engine, get_db
from app.models import LoanPrediction

app = FastAPI(title="Loan Prediction API")

# create tables
Base.metadata.create_all(bind=engine)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health():
    return {"status": "API is running"}


@app.post("/predict", response_model=LoanPredictionResponse)
def predict(payload: LoanPredictionRequest, db: Session = Depends(get_db)):
    try:
        result = predict_loan(payload.model_dump())

        db_record = LoanPrediction(
            income=payload.income,
            credit_score=payload.credit_score,
            loan_amount=payload.loan_amount,
            years_employed=payload.years_employed,
            city=payload.city,
            prediction=result["prediction"],
            decision=result["decision"],
            probability=result["probability"],
        )

        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predictions")
def get_all_predictions(db: Session = Depends(get_db)):
    records = db.query(LoanPrediction).order_by(LoanPrediction.id.desc()).all()
    return records


@app.get("/predictions/{prediction_id}")
def get_prediction_by_id(prediction_id: int, db: Session = Depends(get_db)):
    record = db.query(LoanPrediction).filter(LoanPrediction.id == prediction_id).first()

    if not record:
        raise HTTPException(status_code=404, detail="Prediction record not found")

    return record