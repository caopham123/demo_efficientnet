import cv2, os, time
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

            frame= cv2.resize(frame, (800, 550))
            ### =========== BEGIN PROCESSING FRAME ==============
            # ======   CONFIGURE    ===========
            print(f"\n\n==============\n Frame {frame_order}")
            h, w= frame.shape[:2]
            # display_pic= frame.copy()

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            cv2.imshow('Classify goods', frame)
            # ======   LOAD DETECTION MODEL    ===========
            results = self.detection.model_detection(frame)
            confidience_lst = results[0].boxes.conf
            # ====== FOUND THE BBOX WITH MAX CONFIDIENCE ===========
            if len(confidience_lst) == 0: 
                print("Not found confidiences")
                continue
            
            max_idx= 0
            for idx in range (len(confidience_lst)):
                if (confidience_lst[idx] > confidience_lst[max_idx] 
                    and confidience_lst[idx]> CONF_DETECT_THRESH):
                    max_idx= idx
            max_bbox= results[0].boxes[max_idx]
            x1,y1,x2,y2 = map(int, max_bbox.xyxy[0])

            container_class_id= int(max_bbox.cls[0])
            container_label= self.detection.translate_label(container_class_id)
            conf_score= float(max_bbox.conf[0])

            frame= cv2.rectangle(frame, (x1,y1), (x2,y2),(255,0,100),thickness)
            x1 = max(0, x1 - MARGIN)
            y1 = max(0, y1 - MARGIN)
            x2 = min(w, x2 + MARGIN)
            y2 = min(h, y2 + MARGIN)
            cropped_pic= frame[y1:y2, x1:x2]
            if cropped_pic.size == 0:
                print("Warning: Empty cropped detection!")
                continue


            if container_label== "het hang":
                cv2.putText(frame, container_label, (MARGIN, MARGIN*2), font, font_scale,(255,0,100), thickness)
                cv2.imshow('Classify goods', frame)
                continue

            # print(f"  ======== BEGIN CLASSIFICATION =========")
            # classification_model, class_to_idx = self.classification.load_classification_model()
            # pil_img = Image.fromarray(cv2.cvtColor(cropped_pic, cv2.COLOR_BGR2RGB))
            # input_tensor = self.classification.preprocess_image(pil_img)

            # # Inference image
            # with torch.no_grad():
            #     outputs= classification_model(input_tensor)
            #     probability = nn.functional.softmax(outputs[0], dim=0)
            #     best_prob, best_class = torch.max(probability, 0)
            # # print(f"Best prob: {best_prob} ==== Match class idx: {best_class}")

            # # Get class name and confidence
            # goods_label = self.classification.get_class_name(best_class.item(), class_to_idx)
            # goods_confidence = round(best_prob.item()*100., 2)
            # goods_label = f"{goods_label}: {goods_confidence}%"
            # combined_label = f"{container_label} | {goods_label}"

            # print(f"label: {goods_label}")
            # cv2.putText(frame, combined_label, (MARGIN, MARGIN), fontFace=font
            #                         ,fontScale=font_scale, color=(0, 255, 0), thickness=thickness)
            # print(f"  ======== END CLASSIFICATION =========")

            # # display_pic= cv2.resize(display_pic,  (800, 550))
            # cv2.imshow('Classify goods', frame)

            frame_order+= 1
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        cap.release()
        cv2.destroyAllWindows()

    # def real_time_video(self, video_path, iterval = 0.1):
    #     model_classification, class_to_idx = self.classification.load_classification_model()
    #     cap = cv2.VideoCapture(video_path)
    #     cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    #     frame_order = 0

    #     if not cap.isOpened():
    #         print("Not play video")
    #         return None
        
    #     while cap.isOpened():
    #         last_time = time.perf_counter()
    #         ret, pic = cap.read()
    #         if not ret: break

    #         if frame_order % SKIP_FRAME != 0:
    #             frame_order += 1
    #             continue
                
    #         ### =========== BEGIN PROCESSING FRAME ==============
    #         # ======   CONFIGURE    ===========
    #         print(f"\n\n==============\n Frame {frame_order}")
    #         h, w = pic.shape[:2]
    #         w_begin = int(w/4)
    #         w_end = int(w* 0.9) + MARGIN
    #         print(f"line {w_begin}-{w_end}")
    #         display_pic = pic.copy()
    #         cv2.line(display_pic, (w_begin, 0), (w_begin, h), (0,255,255),2)
    #         cv2.line(display_pic, (w_end, 0), (w_end, h), (0,255,255),2)
    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         font_scale = .7
    #         thickness = 2

    #         # ======   LOAD DETECTION MODEL    ===========
    #         results = self.detection.model_detection(pic)
    #         confidiences = results[0].boxes.conf

    #         # ====== FOUND THE BBOX WITH MAX CONFIDIENCE ===========
    #         max_idx = 0
    #         num_confidiences = len(confidiences)
    #         if num_confidiences == 0: continue

    #         for idx in range (num_confidiences):
    #             print(f"    idx: {idx} - {confidiences[idx]}")
    #             if confidiences[idx] > confidiences[max_idx] and confidiences[idx] > CONF_DETECT_THRESH:
    #                 max_idx = idx
    #         max_bbox = results[0].boxes[max_idx]
    #         x1, y1, x2, y2 = map(int, max_bbox.xyxy[0])
    #         print(f"x1, y1, x2, y2 {x1, y1, x2, y2}")
    #         # =================================
            
    #         # Get class_name for ROI container
    #         cls_id = int(max_bbox.cls[0])
    #         container_label = self.detection.translate_label(cls_id)
    #         conf_detection = float(max_bbox.conf[0])

    #         # Draw bounding box
    #         cv2.rectangle(display_pic, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         x1 = max(0, x1 - MARGIN)
    #         y1 = max(0, y1 - MARGIN)
    #         x2 = min(w, x2 + MARGIN)
    #         y2 = min(h, y2 + MARGIN)
    #         cropped_pic = pic[y1: y2, x1: x2]
    #         if cropped_pic.size == 0: 
    #             print("Warning: Empty crop detected!")
    #             continue            

    #         # ===== PROCESS IN CASE OF EMPTY CONTAINER ==========
    #         if container_label == "het hang":
    #             cv2.putText(display_pic, container_label, (x1, y1), fontFace=font, fontScale=font_scale, color=(0, 255, 0), thickness=thickness) 
    #             continue

    #         # ===== PROCESS IN CASE OF NOT EMPTY CONTAINER ==========
    #         # Requirement:
    #         # 1. If edges's container over line w_begin and w_end
    #         # Beginning Image Classification
    #         # 2. After classification, hold 6-10 frames to get certain result (label_score)
    #         if x1 >= w_begin and x2 >= w_end : 
    #             print(f"  ======== BEGIN CLASSIFICATION =========")
    #             pil_img = Image.fromarray(cv2.cvtColor(cropped_pic, cv2.COLOR_BGR2RGB))
    #             input_tensor = self.classification.preprocess_image(pil_img)
    #             # Inference
    #             with torch.no_grad():
    #                 outputs = model_classification(input_tensor)
    #                 probability = nn.functional.softmax(outputs[0], dim=0)
    #                 best_prob, best_class = torch.max(probability, 0)

    #             # Get class name and confidence
    #             goods_name = self.classification.get_class_name(best_class.item(), class_to_idx)
    #             conf_goods = round(best_prob.item()*100., 2)
    #             goods_label = f"{goods_name}: {conf_goods}%" if container_label != "het hang" else ""
    #             print(f"label_classification: {goods_label}")

    #             # Display prediction on frame
    #             (label_w,_), _ = cv2.getTextSize(goods_label, font, font_scale, thickness)
    #             cv2.putText(display_pic, container_label, (x1, y1), fontFace=font, fontScale=font_scale, color=(0, 255, 0), thickness=thickness)
    #             cv2.putText(display_pic, goods_label, (x2 - label_w, y1), fontFace=font, fontScale=font_scale, color=(0, 255, 0), thickness=thickness)
    #             print(f"  ======== END CLASSIFICATION =========")
    #         # ===========================
            
    #         # Display the resulting frame
    #         resized_pic = cv2.resize(display_pic, (800, 600))
    #         cv2.imshow('Classify goods', resized_pic)

    #         ## After for loop
    #         process_time = time.perf_counter() - last_time
    #         remain_time = max(0, iterval - process_time)
    #         print(f"pro_time: {process_time}")
    #         print(f"remain_time: {remain_time}")

    #         if remain_time > 0:
    #             key = cv2.waitKey(int(remain_time*1000))
    #         else:
    #             key = cv2.waitKey(1)
            
    #         if key == ord('q'): break
    #         frame_order += 1
    #     ### ========== END PROCESSING FRAME =============
    #     cap.release()
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    goods_recognizer = GoodsRecognition()
    # goods_recognizer.recognize_video_goods("rtsp://admin:longson2016@192.168.10.26:1555/Streaming/Channels/5001")
    # goods_recognizer.recognize_video_goods("D:/LONGSON/PROJECTS/demo_efficientnet/data/25_07_21_08_30.mp4")
    goods_recognizer.recognize_video_goods("D:/LONGSON/PROJECTS/demo_efficientnet/data/25_07_21_08_31.mp4")