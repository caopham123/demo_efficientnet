from fastapi.responses import JSONResponse
from fastapi import status
from datetime import datetime
import os
from ai_core.src.goods_classification import GoodsClassification
from ai_core.src.container_detection import ContainerDetection
from ai_core.src.setting import RESULT_PATH
from api.helpers.db_connection import QueryMember
from api.helpers.commons import stringToRGB, RGB2String


goods_classifier= GoodsClassification()
query_member= QueryMember()

class PredictionService:
    def __init__(self):
        self.goods_classifier= GoodsClassification()
        self.query_member= QueryMember()
        self.container_detection= ContainerDetection()

    def create_prediction(self, img_str: str):
        if img_str is None:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                                content={"message":"Hinh anh khong hop le!!!"})
        
        detection_result= self.container_detection.detect_container_by_path(img_str)
        if detection_result is None:
            print("======= Not found detection")
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                                content={"message":"Khong phat hien duoc vat the!!!"})

        display_pic, goods_label_lst, goods_score_lst= self.goods_classifier.classify_goods_img(img_str)
        converted_pic= RGB2String(display_pic)
        created_at= datetime.now()

        result= self.query_member.create_new_prediction(product=goods_label_lst[0], score=goods_score_lst[0], image=converted_pic)
        if result:
            return JSONResponse(status_code=status.HTTP_201_CREATED, content={"message":f"Nhan dien thanh cong {goods_label_lst[0]} luc {created_at}"})
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": result})
            
