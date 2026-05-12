from pydantic import BaseModel


class PredictionResult(BaseModel):
    rank: int
    class_name: str
    confidence: float


class PredictionResponse(BaseModel):
    top_predictions: list[PredictionResult]
    predicted_class: str
    confidence: float
    processing_time_ms: float


class BatikInfoResponse(BaseModel):
    class_name: str
    origin: str
    region: str
    description: str
    characteristics: list[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
