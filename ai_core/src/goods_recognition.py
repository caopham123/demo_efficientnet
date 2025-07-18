import cv2, os
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image   
from setting import *
from container_detection import ContainerDetection
from goods_classification import GoodsClassification
os.makedirs(RESULT_PATH, exist_ok=True)

class GoodsRecognition:
    def __init__(self):
        self.detection = ContainerDetection()
        self.classification = GoodsClassification()

    def recognize_video_goods(self, video_path):
        cap= cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_order= 0

        if not cap.isOpened():
            return None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                print("=========> Err during read frame!")
                break
            # cv2.imshow("frame of cap", frame)
            if frame_order% SKIP_FRAME !=0:
                frame_order +=1
                continue

            h, w= frame.shape[:2]
            display_pic= frame.copy()
            # ======   LOAD DETECTION MODEL    ===========
            results = self.detection.model_detection(frame)
            if results is None:
                print("Not found detection")
                continue

            confidience_lst = results[0].boxes.conf
            if len(confidience_lst) == 0: 
                print("Not found confidiences")
                continue
            
            print(f"Found {len(confidience_lst)} confidience")
            max_idx= 0
            for idx in range (len(confidience_lst)):
                if (confidience_lst[idx] > confidience_lst[max_idx] 
                    and confidience_lst[idx]> CONF_DETECT_THRESH):
                    max_idx= idx
            
            max_bbox= results[0].boxes[max_idx]
            x1,y1,y2,x2 = map(int, max_bbox.xyxy[0])
            class_id= int(max_bbox.cls[0])
            class_name= self.translate_label(class_id)
            conf_score= float(max_bbox.conf[0])
            conf_score= round(conf_score* 100, 2)

            x1 = max(0, x1 - MARGIN)
            y1 = max(0, y1 - MARGIN)
            x2 = min(w, x2 + MARGIN)
            y2 = min(h, y2 + MARGIN)
            cropped_pic= frame[y1-MARGIN : y2+MARGIN, x1-MARGIN : x2+MARGIN]
            if cropped_pic.size == 0:
                continue

            display_pic = cv2.rectangle(frame, (x1, y1), (x2, y2), (100,50,200), 1)



            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    goods_recognizer = GoodsRecognition()
    goods_recognizer.recognize_video_goods("rtsp://admin:longson2016@192.168.10.26:1555/Streaming/Channels/5002")
    # goods_recognizer.recognize_video_goods("D:/LONGSON/PROJECTS/demo_efficientnet/data/25_07_16_06_48.mp4")