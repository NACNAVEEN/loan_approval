from pydantic import BaseModel, Field


class LoanPredictionRequest(BaseModel):
    income: int = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=850)
    loan_amount: int = Field(..., gt=0)
    years_employed: int = Field(..., ge=0)
    city: str