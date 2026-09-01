from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/house_price_model.joblib")

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0"
)

model = None


class HouseFeatures(BaseModel):
    area: float = Field(..., gt=0)
    bedrooms: int = Field(..., ge=0)
    city: str
    furnishing: str | None = None
    locality_tier: str | None = None


@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Run training first."
        )
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    row = pd.DataFrame([features.model_dump()])
    prediction = float(model.predict(row)[0])

    return {
        "predicted_price_lakhs": round(prediction, 4)
    }
