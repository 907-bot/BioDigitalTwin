from app.core.database import Base
from sqlalchemy import Column, String, Integer, Float, DateTime
from sqlalchemy.sql import func


class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(String, primary_key=True, index=True)
    age = Column(Integer, nullable=False)
    gender = Column(String, nullable=False)
    bmi = Column(Float, nullable=False)
    hr = Column(Integer, nullable=False)
    hrv = Column(Integer, nullable=False)
    spo2 = Column(Float, nullable=False)
    glucose = Column(Integer, nullable=False)
    systolic_bp = Column(Integer, nullable=False)
    diastolic_bp = Column(Integer, nullable=False)
    embedding = Column(String, nullable=True)
    risk_score = Column(Float, nullable=True)
    risk_label = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
