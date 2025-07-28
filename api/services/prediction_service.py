from ai_core.src.goods_classification import GoodsClassification
from fastapi.responses import JSONResponse
from fastapi import status
from api.helpers.db_connection import QueryMember
from datetime import datetime

goods_classifier= GoodsClassification()
query_member= QueryMember()

class PredictionService:
    def __init__(self):
        self.goods_classifier= GoodsClassification()
        self.query_member= QueryMember()

    # def create_prediction(self, )