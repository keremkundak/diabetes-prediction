import sys
sys.path.append('src')

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from feature_engineering import create_features

app = FastAPI()

# Input data model
class BloodValues(BaseModel):
    Age: int
    Gender: str
    BMI: float
    Chol: float
    TG: float
    HDL: float
    LDL: float
    Cr: float
    BUN: float

# Modeli ve scaler'ı yükle
model = joblib.load('models/random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.post("/predict")
def predict(data: BloodValues):
    # Inputları dataframe'e çevir
    gender_map = {'Erkek': 1, 'Kadın': 0}
    input_data = pd.DataFrame({
        'Age': [data.Age],
        'Gender': [gender_map[data.Gender]],
        'BMI': [data.BMI],
        'Chol': [data.Chol],
        'TG': [data.TG],
        'HDL': [data.HDL],
        'LDL': [data.LDL],
        'Cr': [data.Cr],
        'BUN': [data.BUN]
    })

    # Feature engineering
    input_data = create_features(input_data)

    # Veriyi ölçeklendir
    numerical_cols = [col for col in input_data.columns if input_data[col].dtype != 'object']
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

    # Tahmin yap
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    return {
        "prediction": int(prediction[0]),
        "prediction_probability": float(prediction_proba[0][1])
    }