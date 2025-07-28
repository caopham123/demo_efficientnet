from api.schema import PredictionMaking, PredictionModification
from api.services.prediction_service import *
from api.helpers.commons import stringToRGB
from fastapi import status, APIRouter

router = APIRouter(
    prefix="/api/v1",
    tags=["GOODS RECOGNITION"],
    responses={}
)
    
@router.get("/ping")
async def ping():
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message":"pong"})