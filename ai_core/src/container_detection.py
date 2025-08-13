from ultralytics import YOLO
import cv2, os, numpy
import torch, timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image   
from ai_core.src.setting import MODEL_DETECTION, CONF_DETECT_THRESH, RESULT_PATH
from api.helpers.commons import stringToRGB

MARGIN = 20
model_file = os.path.join(MODEL_DETECTION, "best.pt")

class ContainerDetection:
    def __init__(self):
        self.model_detection = YOLO(model_file.replace('\\', '/'))

    def translate_label(self, idx_class):
        vn_lst_name = ["het hang", "con hang"]
        return vn_lst_name[idx_class]

    def detect_container_by_img(self, img_input):
        if img_input is None: 
            print("Not found input img")
            return None
        
        pic = img_input
        print(type(pic))

        results = self.model_detection(pic)
        if results[0] is None:
            print("Not found detection")
            # return None

        confidence_lst = results[0].boxes.conf
        if len(confidence_lst) == 0: 
            print("Not found confidences")
            return None
        
        print(f"Found {len(confidence_lst)} confidence")
        max_idx= 0
        for idx in range (len(confidence_lst)):
            if (confidence_lst[idx] > confidence_lst[max_idx] 
                and confidence_lst[idx]> CONF_DETECT_THRESH):
                max_idx= idx
        
        max_bbox= results[0].boxes[max_idx]
        x1,y1,y2,x2 = map(int, max_bbox.xyxy[0])
        class_id= int(max_bbox.cls[0])
        class_name= self.translate_label(class_id)
        conf_score= float(max_bbox.conf[0])
        conf_score= round(conf_score* 100, 2)

        cropped_pic= pic[y1-MARGIN : y2+MARGIN, x1-MARGIN : x2+MARGIN]
        if cropped_pic.size == 0:
            return None
        display_pic = cv2.rectangle(pic, (x1, y1), (x2, y2), (100,50,200), 1)
        cropped_pic = cv2.resize(cropped_pic, (600, 400))
        return display_pic, cropped_pic, class_name

    def detect_container_by_path(self, img_str:str):
        # pic= cv2.imread(img_path.replace("\\","/"))
        pic= stringToRGB(img_str)
        pic= cv2.resize(pic, (800, 550))
        h, w= pic.shape[:2]
        results = self.model_detection(pic)
        if results is None:
            print("Not found detection")
            return None

        display_pic= pic.copy()
        cropped_pic_lst= []
        container_label_lst= []
        container_score_lst= []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1= max(0, x1-MARGIN)
            y1= max(0, y1-MARGIN)
            x2= max(w, x2+MARGIN)
            y2= max(h, y2+MARGIN)

            class_id = int(box.cls[0]) # Get class ID: 0, 1, 2
            class_name = self.translate_label(class_id) # Get class name & score
            conf = float(box.conf[0])
            conf = round(conf*100., 2)

            display_pic= cv2.rectangle(pic, (x1, y1), (x2, y2), (0,255,0), 2)
            cropped_pic= pic[y1-MARGIN : y2+MARGIN, x1-MARGIN : x2+MARGIN]

            cropped_pic_lst.append(cropped_pic)
            container_label_lst.append(class_name)
            container_score_lst.append(conf)

        return display_pic, cropped_pic_lst, container_label_lst, container_score_lst

if __name__ == "__main__":
    container = ContainerDetection()
    container.detect_container_by_path("D:\LONGSON\PROJECTS\demo_efficientnet\data\img15.jpg")
