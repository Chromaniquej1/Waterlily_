from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load

# Load model (adjust path if needed)
model = load("/Users/jayantbiradar/Desktop/Waterlily-coding/GradientBoosting_best_model.joblib")

app = FastAPI(title="Diabetes Readmission API")

# Request schema (matches model features)
class Encounter(BaseModel):
    race: str
    gender: str
    age: str
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    A1Cresult: str = None
    max_glu_serum: str = None
    change: str
    diabetesMed: str
    insulin: str

@app.get("/health")
def health():
    return {"status": "Ok"}

@app.post("/predict")
def predict(records: list[Encounter]):
    data = pd.DataFrame([r.dict() for r in records])
    probs = model.predict_proba(data)[:, 1]
    preds = model.predict(data)
    return {"predictions": preds.tolist(), "probabilities": probs.tolist()}
