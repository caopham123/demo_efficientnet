from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class PredictionMaking2(BaseModel):
    product: str= Field(...,  example="coal")
    score: float= Field(...,  example=80.)
    created_at: datetime= Field(...,  example='2021-09-02 07:00:00')
    modified_at: datetime= Field(...,  example='2021-09-02 07:00:00')
    is_delete: bool= Field(...,  example=False)
    image: str= Field(...,  example="base64")
    class Config:
        json_schema_extra = {
            "product": "coal",
            "score": 80.,
            "created_at": '2021-09-02 07:00:00',
            "modified_at": '2021-09-02 07:00:00',
            'is_delete': False,
            "image": "base64"
        }

class Image(BaseModel):
    image: str = Field(..., example="base64-encoded-image")

class PredictionModification(BaseModel):
    product: Optional[str]
    score: Optional[float]
    created_at: Optional[datetime]
    modified_at: Optional[datetime]
    is_delete: Optional[bool]
    image: Optional[str]
    class Config:
        json_schema_extra = {
            "product": "coal",
            "score": 80.,
            "created_at": '2021-09-02 07:00:00',
            "modified_at": '2021-09-02 07:00:00',
            'is_delete': False,
            "image": "base64"
        }
