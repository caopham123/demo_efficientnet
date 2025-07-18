from ultralytics import YOLO
import cv2, os, numpy
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image   
import time
from setting import MODEL_DETECTION, CONF_DETECT_THRESH, RESULT_PATH

MARGIN = 20
model_file = os.path.join(MODEL_DETECTION, "best.pt")

class ContainerDetection:
    def __init__(self):
        self.model_detection = YOLO(model_file.replace('\\', '/'))

    def translate_label(self, idx_class):
        vn_lst_name = ["het hang", "con hang"]
        return vn_lst_name[idx_class]

    def detect_container(self, img_input):
        if img_input is None: 
            print("Not found input img")
            return None
        
        pic = img_input
        print(type(pic))

        results = self.model_detection(pic)
        if results[0] is None:
            print("Not found detection")
            # return None

        confidience_lst = results[0].boxes.conf
        if len(confidience_lst) == 0: 
            print("Not found confidiences")
            return None
        
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

        cropped_pic= pic[y1-MARGIN : y2+MARGIN, x1-MARGIN : x2+MARGIN]
        if cropped_pic.size == 0:
            return None
        display_pic = cv2.rectangle(pic, (x1, y1), (x2, y2), (100,50,200), 1)
        cropped_pic = cv2.resize(cropped_pic, (600, 400))
        return display_pic, cropped_pic, class_name

    def detect_container_2(self, img_path):
        pic = cv2.imread(img_path)

        results = self.model_detection(pic)
        if results is None:
            print("Not found detection")
            return None

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class ID: 0, 1, 2
            class_id = int(box.cls[0]) 
            # Get class name & score
            class_name = self.translate_label(class_id)
            conf = float(box.conf[0])
            conf = round(conf*100., 2)

            display_pic = cv2.rectangle(pic, (x1, y1), (x2, y2), (0,255,0), 2)
            cropped_pic = pic[y1-MARGIN : y2+MARGIN, x1-MARGIN : x2+MARGIN]

        cropped_pic = cv2.resize(cropped_pic, (600, 400))
        return display_pic, cropped_pic, class_name


# def detect_container_video(url):
#     cap = cv2.VideoCapture(url)
#     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     frame_counter = 0
#     while cap.isOpened():
#         ret, pic = cap.read()
#         if not ret: break
#         frame_counter += 1
#         results = model_detection(pic)
#         masks = results[0].boxes.conf > CONF_DETECT_THRESH
#         filtered_conf = results[0].boxes[masks]
#         for i, box in enumerate(filtered_conf):
#             print(f"    ======\n  Box conf {i}. of frame {frame_counter}")
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             print(f"x1, y1, x2, y2 {x1, y1, x2, y2}")
#             # Get class ID
#             cls_id = int(box.cls[0])

#             # Get class name
#             cls_name = translate_label(cls_id)

#             # Optional: Get confidence score
#             conf = float(box.conf[0])

#             # Draw bounding box
#             cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cropped_pic = pic[y1 -MARGIN: y2 +MARGIN, x1 -MARGIN: x2 +MARGIN]
#             # Draw label
#             label = f"{cls_name} {conf:.2f}"
#             cv2.putText(pic, label, (x1, y1 + 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#         resized_pic = cv2.resize(pic, (500, 400))
#         cv2.imshow("Container Detection", resized_pic)
#         if cv2.waitKey(25) & 0xFF == ord('q'): break

#     cap.release()
#     cv2.destroyAllWindows()
#     # return resized_pic, cropped_pic, cls_name

if __name__ == "__main__":
    container = ContainerDetection()
    # real_time_video("rtsp://admin:longson2016@192.168.10.26:1555/Streaming/Channels/5002")
    # real_time_video("./data/25_07_16_12_5.mp4")