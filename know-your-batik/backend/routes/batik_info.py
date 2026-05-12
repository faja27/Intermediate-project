from fastapi import APIRouter, HTTPException, Request

from backend.models.predictor import BATIK_INFO
from backend.schemas.request_response import BatikInfoResponse

router = APIRouter()


@router.get("/batik/{class_name}", response_model=BatikInfoResponse)
async def get_batik_info(class_name: str, request: Request):
    predictor = request.app.state.predictor
    info = predictor.get_class_info(class_name)
    if not info:
        raise HTTPException(status_code=404, detail=f"Class '{class_name}' not found")
    return BatikInfoResponse(**info)


@router.get("/classes", response_model=list[str])
async def list_classes():
    return sorted(BATIK_INFO.keys())
