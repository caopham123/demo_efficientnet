from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class PredictionMaking(BaseModel):
    product: str= Field(examples="coal")
    score: float= Field(examples=80.)
    created_at: datetime= Field(examples='2021-09-02 07:00:00')
    modified_at: datetime= Field(examples='2021-09-02 07:00:00')
    is_delete: bool= Field(examples=False)
    image: str= Field(examples='base64')
    class Config:
        json_schema_extra = {
            "product": "coal",
            "score": 80.,
            "created_at": '2021-09-02 07:00:00',
            "modified_at": '2021-09-02 07:00:00',
            'is_delete': False,
            "image": "base64"
        }

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
