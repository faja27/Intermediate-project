import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is on sys.path so src/ and backend/ are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import DEVICE, LABELS_PATH, MODEL_PATH, NUM_CLASSES
from backend.models.predictor import BatikPredictor
from backend.routes import batik_info, predict
from backend.schemas.request_response import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = BatikPredictor(
        model_path=str(MODEL_PATH),
        labels_path=str(LABELS_PATH),
        device=DEVICE,
    )
    print(f"Model loaded on {DEVICE} — {NUM_CLASSES} classes ready.")
    yield
    del app.state.predictor


app = FastAPI(title="Know Your Batik API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(batik_info.router)


@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = hasattr(app.state, "predictor") and app.state.predictor is not None
    return HealthResponse(status="ok", model_loaded=loaded, num_classes=NUM_CLASSES)


@app.get("/")
async def root():
    return {"message": "Know Your Batik API", "docs": "/docs"}
