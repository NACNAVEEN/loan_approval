from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from datetime import datetime
from app.database import Base


class LoanPrediction(Base):
    __tablename__ = "loan_predictions"

    id = Column(Integer, primary_key=True, index=True)
    income = Column(Integer, nullable=False)
    credit_score = Column(Integer, nullable=False)
    loan_amount = Column(Integer, nullable=False)
    years_employed = Column(Integer, nullable=False)
    city = Column(String(100), nullable=False)

    prediction = Column(Boolean, nullable=False)
    decision = Column(String(20), nullable=False)
    probability = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)