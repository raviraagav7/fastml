from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import joblib
import pandas as pd

app = FastAPI()

class ScoringItem(BaseModel):
    gender: Literal['Male', 'Female']
    SeniorCitizen: int  # 0 or 1
    Partner: Literal['Yes', 'No']
    Dependents: Literal['Yes', 'No']
    tenure: int
    PhoneService: Literal['Yes', 'No']
    MultipleLines: Literal['Yes', 'No', 'No phone service']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: Literal['Yes', 'No', 'No internet service']
    OnlineBackup: Literal['Yes', 'No', 'No internet service']
    DeviceProtection: Literal['Yes', 'No', 'No internet service']
    TechSupport: Literal['Yes', 'No', 'No internet service']
    StreamingTV: Literal['Yes', 'No', 'No internet service']
    StreamingMovies: Literal['Yes', 'No', 'No internet service']
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: Literal['Yes', 'No']
    PaymentMethod: Literal[
        'Electronic check', 
        'Mailed check', 
        'Bank transfer (automatic)', 
        'Credit card (automatic)'
    ]
    MonthlyCharges: float
    TotalCharges: float

model = joblib.load("churn_model.pkl")

@app.post('/')
async def scoring_endpoint(item: ScoringItem):

    df = pd.DataFrame([item.dict()])
    pred = model.predict(df)[0]
    return {"prediction": int(pred)}