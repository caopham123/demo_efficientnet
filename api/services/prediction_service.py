from ai_core.src.goods_classification import GoodsClassification
from fastapi.responses import JSONResponse
from fastapi import status
from api.helpers.db_connection import QueryMember
from datetime import datetime
from api.helpers.commons import stringToRGB, RGB2String

goods_classifier= GoodsClassification()
query_member= QueryMember()

class PredictionService:
    def __init__(self):
        self.goods_classifier= GoodsClassification()
        self.query_member= QueryMember()

    def create_prediction(self, img_str: str):
        if img_str is None:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                                content={"message":"Hinh anh khong hop le!!!"})
        # img_input= stringToRGB(img_str)
        display_pic, goods_label_lst, goods_score_lst= self.goods_classifier.classify_goods_img(img_str)

        converted_pic= RGB2String(display_pic)
        result= self.query_member.create_new_prediction(product=goods_label_lst[0], score=goods_score_lst[0], image=converted_pic)
        created_at= datetime.now()
        if result:
            return JSONResponse(status_code=status.HTTP_201_CREATED, content={"message":f"Nhan dien thanh cong luc {created_at}"})
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"message": result})
        
