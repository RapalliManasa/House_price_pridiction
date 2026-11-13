from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# Load saved pipeline
model = joblib.load("dt_pipeline.pkl")

app = FastAPI(title="House Price Prediction API")


# Input Schema
class PropertyInput(BaseModel):
    City: str
    Locality_Type: str
    Distance_to_Metro_km: float
    Distance_to_City_Center_km: float
    Property_Type: str
    BHK: int
    Bathrooms: int
    Super_Area_SqFt: float
    Carpet_Area_SqFt: float
    Floor_Number: int
    Total_Floors: int
    Age_of_Property: int
    Furnishing_Status: str
    Parking: int
    Lift_Available: int
    Gated_Community: int
    Balcony_Count: int
    Water_Supply: str
    Power_Backup: str
    Security: int
    Gym_Available: int
    Swimming_Pool: int
    Clubhouse: int
    Children_Play_Area: int
    Park_Nearby: int
    Hospital_Distance_km: float
    School_Distance_km: float
    Shopping_Mall_Distance_km: float
    Road_Width_ft: int
    Facing: str
    Ownership_Type: str
    Construction_Status: str
    Transaction_Type: str
    Nearby_Metro: int
    Near_Main_Road: int
    Air_Quality_Index: int
    Noise_Level_dB: float
    Green_Cover_Percent: float
    Property_Tax_Annual_INR: int
    Maintenance_Monthly_INR: int
    Price_Per_SqFt_INR: float


# Prediction endpoint
@app.post("/predict")
def predict(data: PropertyInput):

    # Convert input into DataFrame
    df = pd.DataFrame([data.dict()])

    # Predict house price
    pred = model.predict(df)[0]

    # Convert log prediction to original price
    price = np.expm1(pred)

    return {
        "predicted_price_in_lakhs": round(float(price), 2),
        "predicted_price_in_INR": round(float(price * 100000), 2)
    }