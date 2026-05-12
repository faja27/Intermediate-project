import io
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from PIL import Image

from backend.config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB
from backend.schemas.request_response import PredictionResponse, PredictionResult

router = APIRouter()

_MAX_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    contents = await file.read()
    if len(contents) > _MAX_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE_MB} MB",
        )

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image file")

    predictor = request.app.state.predictor
    t0 = time.perf_counter()
    raw = predictor.predict(image)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    top_predictions = [
        PredictionResult(rank=r["rank"], class_name=r["class"], confidence=r["confidence"])
        for r in raw
    ]
    best = top_predictions[0]

    return PredictionResponse(
        top_predictions=top_predictions,
        predicted_class=best.class_name,
        confidence=best.confidence,
        processing_time_ms=round(elapsed_ms, 2),
    )
