# from pydantic import BaseModel, Field


# class LoanPredictionRequest(BaseModel):
#     income: int = Field(..., gt=0)
#     credit_score: int = Field(..., ge=300, le=850)
#     loan_amount: int = Field(..., gt=0)
#     years_employed: int = Field(..., ge=0)
#     city: str

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class LoanPredictionRequest(BaseModel):
    income: int = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: int = Field(..., gt=0)
    years_employed: int = Field(..., ge=0)
    city: str


class LoanPredictionResponse(BaseModel):
    prediction: bool
    decision: str
    probability: Optional[float]


class LoanPredictionHistory(BaseModel):
    id: int
    income: int
    credit_score: int
    loan_amount: int
    years_employed: int
    city: str
    prediction: bool
    decision: str
    probability: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True