import asyncio
import contextlib

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REGISTERED_MODEL_NAME = "house_price_model"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
REFRESH_INTERVAL_SECONDS = 10  # matches the 10-minute retrain schedule

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0"
)

model = None
_model_version_info = None
_refresh_task = None


class HouseFeatures(BaseModel):
    area: float = Field(..., gt=0)
    bedrooms: int = Field(..., ge=0)
    city: str
    furnishing: str | None = None
    locality_tier: str | None = None


def _load_production_model():
    global model, _model_version_info
    client = mlflow.MlflowClient()
    version_info = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "production")

    # Only reload the actual model object if the production version changed
    if _model_version_info is None or version_info.version != _model_version_info.version:
        model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@production")
        print(f"Loaded production model version {version_info.version}")

    _model_version_info = version_info


async def _refresh_loop():
    while True:
        await asyncio.sleep(REFRESH_INTERVAL_SECONDS)
        try:
            _load_production_model()
        except Exception as e:
            print(f"Model refresh failed: {e}")


@app.on_event("startup")
def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _load_production_model()

    global _refresh_task
    _refresh_task = asyncio.create_task(_refresh_loop())


@app.on_event("shutdown")
async def shutdown():
    if _refresh_task:
        _refresh_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _refresh_task


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": REGISTERED_MODEL_NAME,
        "version": _model_version_info.version,
        "alias": "production",
        "run_id": _model_version_info.run_id,
    }


@app.post("/predict")
def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    row = pd.DataFrame([features.model_dump()])
    prediction = float(model.predict(row)[0])

    return {
        "predicted_price_lakhs": round(prediction, 4)
    }