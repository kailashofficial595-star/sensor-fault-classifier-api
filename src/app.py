import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import FEATURES, add_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fault-api")

MODEL_PATH = Path("models/model.joblib")
artifact = joblib.load(MODEL_PATH)  # {"model", "threshold", "columns"}
model = artifact["model"]
threshold = artifact["threshold"]
columns = artifact["columns"]

app = FastAPI(title="Sensor Fault Classifier API", version="1.0.0")


class SensorInput(BaseModel):
    air_temp: float = Field(..., gt=200, lt=400, description="Air temperature in Kelvin")
    process_temp: float = Field(..., gt=200, lt=400, description="Process temperature in Kelvin")
    rpm: float = Field(..., gt=0, lt=10000, description="Rotational speed in rpm")
    torque: float = Field(..., ge=0, lt=500, description="Torque in Nm")
    tool_wear: float = Field(..., ge=0, lt=1000, description="Tool wear in minutes")

    model_config = {
        "json_schema_extra": {
            "example": {"air_temp": 298.1, "process_temp": 308.6, "rpm": 1551,
                        "torque": 42.8, "tool_wear": 0}
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(data: SensorInput):
    row = pd.DataFrame([data.model_dump()])[FEATURES]
    x = add_features(row)[columns]  # same feature code as training
    probability = float(model.predict_proba(x)[0, 1])
    fault = probability >= threshold
    logger.info("prediction fault_probability=%.3f fault=%s input=%s",
                probability, fault, data.model_dump())
    return {"fault_probability": round(probability, 4), "fault": fault,
            "threshold": round(float(threshold), 4)}