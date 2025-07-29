import cv2, os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image   
from ai_core.src.setting import *
from ai_core.src.container_detection import ContainerDetection
from ai_core.src.goods_classification import GoodsClassification
os.makedirs(RESULT_PATH, exist_ok=True)

class GoodsRecognition:
    def __init__(self):
        self.detection = ContainerDetection()
        self.classification = GoodsClassification()

    def recognize_video_goods(self, video_path):
        classification_model, class_to_idx = self.classification.load_classification_model()
        classification_model.eval()

        cap= cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        frame_order= 0

        if not cap.isOpened():
            return None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            if frame_order% SKIP_FRAME !=0:
                frame_order +=1
                continue

            ### =========== BEGIN PROCESSING FRAME ==============
            # ======   CONFIGURE    ===========
            print(f"\n\n==============\n Frame {frame_order}")
            frame= cv2.resize(frame, (800, 550))
            h, w= frame.shape[:2]
            display_frame= frame.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2

            # ======   CONTAINER DETECTION MODEL    ===========
            results = self.detection.model_detection(frame)
            confidence_lst = results[0].boxes.conf
            valid_idx= [i for i, conf in enumerate(confidence_lst) if conf > CONF_DETECT_THRESH]
            if not valid_idx:
                print("No detections found!")
                cv2.imshow('Classify goods', display_frame)
                frame_order+= 1
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            max_idx= valid_idx[0]
            for idx in valid_idx:
                print(f"valid_idx: {idx}")
                if (confidence_lst[idx] >= confidence_lst[max_idx] ):
                    max_idx= idx
            max_bbox= results[0].boxes[max_idx]
            x1,y1,x2,y2 = map(int, max_bbox.xyxy[0])

            x1 = max(0, x1 - MARGIN)
            y1 = max(0, y1 - MARGIN)
            x2 = min(w, x2 + MARGIN)
            y2 = min(h, y2 + MARGIN)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 100), thickness)

            container_class_id= int(max_bbox.cls[0])
            container_label= self.detection.translate_label(container_class_id)
            conf_score= float(max_bbox.conf[0])
            
            cropped_pic= frame[y1:y2, x1:x2]
            if cropped_pic.size == 0:
                print("Warning: Empty cropped detection!")
                cv2.imshow('Classify goods', display_frame)
                frame_order+= 1
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            if container_label== "het hang":
                print(f"container_label: {container_label}")
                cv2.putText(display_frame, container_label, (MARGIN, MARGIN*2), font, font_scale,(0, 255, 0), thickness)
                cv2.imshow('Classify goods', display_frame)
                frame_order+= 1
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            print(f"  ======== BEGIN CLASSIFICATION =========")
            pil_img = Image.fromarray(cv2.cvtColor(cropped_pic, cv2.COLOR_BGR2RGB))
            input_tensor = self.classification.preprocess_image(pil_img)

            # Inference image
            with torch.no_grad():
                outputs= classification_model(input_tensor)
                probability = nn.functional.softmax(outputs[0], dim=0)
                best_prob, best_class = torch.max(probability, 0)

            # Get class name and confidence
            goods_label = self.classification.get_class_name(best_class.item(), class_to_idx)
            goods_confidence = round(best_prob.item()*100., 2)
            goods_label = f"{goods_label}: {goods_confidence}%"
            combined_label = f"{container_label} | {goods_label}"

            print(f"label: {goods_label}")
            cv2.putText(display_frame, combined_label, (MARGIN, MARGIN*2), fontFace=font
                                    ,fontScale=font_scale, color=(0, 255, 0), thickness=thickness)
            cv2.imshow('Classify goods', display_frame)
            print(f"  ======== END CLASSIFICATION =========")

            frame_order+= 1
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    goods_recognizer = GoodsRecognition()
    # goods_recognizer.recognize_video_goods("rtsp://admin:longson2016@192.168.10.26:1555/Streaming/Channels/5001")
    # goods_recognizer.recognize_video_goods("D:/LONGSON/PROJECTS/demo_efficientnet/data/25_07_21_08_30.mp4")
    goods_recognizer.recognize_video_goods("D:/LONGSON/PROJECTS/demo_efficientnet/data/25_07_21_08_31.mp4")