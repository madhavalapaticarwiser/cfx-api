from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .model_utils import CarPriceEnsemble

ROOT = Path("/app")
MODEL_PATHS = {
    "Retail":   str(ROOT / "car_price_model_retail.pkl"),
    "Private":  str(ROOT / "car_price_model_private.pkl"),
    "Trade-In": str(ROOT / "car_price_model_tradein.pkl"),
}
DATA_PATH = str(ROOT / "CFXCLEANEDDATA081225.csv")

app = FastAPI(title="CFX_API (Retail / Private / Trade-In)")

ensemble = CarPriceEnsemble(MODEL_PATHS, DATA_PATH)

class CarInput(BaseModel):
    year: int = Field(..., example=2020, ge=1900)
    mileage: int = Field(..., example=55000, ge=0)

    make: str = Field(..., example="Toyota")
    model: str = Field(..., example="Sienna")
    trim: str = Field(..., example="LE")

    interior: str = Field(..., example="great")
    exterior: str = Field(..., example="great")
    mechanical: str = Field(..., example="great")

    line: str = Field(..., example="Economy")
    drivetrain: str = Field(..., example="FWD")
    transmission: str = Field(..., example="5-speed automatic")

    model_config = {
        "json_schema_extra": {
            "example": {
                "year": 2020,
                "mileage": 55000,
                "make": "string",
                "model": "string",
                "trim": "string",
                "interior": "string",
                "exterior": "string",
                "mechanical": "string",
                "line": "string",
                "drivetrain": "string",
                "transmission": "string"
            }
        }
    }

class PredictionResponse(BaseModel):
    success: bool
    predictions: dict | None = None
    matched_vehicle: dict | None = None
    message: str | None = None

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CarInput):
    try:
        out = ensemble.predict_all(payload.dict(), enforce_gap=True)
        return PredictionResponse(success=True, **out)
    except ValueError as e:
        return PredictionResponse(success=False, message=str(e))
