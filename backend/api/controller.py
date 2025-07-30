from api.schema import Image, PredictionModification
from api.services.prediction_service import PredictionService
from api.helpers.commons import stringToRGB
from fastapi import status, APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(
    prefix="/api/v1",
    tags=["GOODS RECOGNITION"],
    responses={}
)

prediction= PredictionService()
@router.get("/ping")
async def ping():
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message":"pong"})

@router.post("/predict")
async def create_new_predict(item: Image):
    return prediction.create_prediction(item.image)