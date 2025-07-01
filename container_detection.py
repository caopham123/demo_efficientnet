from ultralytics import YOLO
import cv2, os, numpy
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image   
import time
from setting import MODEL_DETECTION, CONF_DETECT_THRESH, SKIP_FRAME
from goods_classification import load_model_classification, get_class_name, preprocess_image

MARGIN = 20
model_file = os.path.join(MODEL_DETECTION, "best.pt")

model_detection = YOLO(model_file.replace('\\', '/'))
class_names = model_detection.names


def translate_label(idx_class):
    vn_lst_name = ["het hang", "con hang"]

    return vn_lst_name[idx_class]


def detect_container_img(url):
    pic = cv2.imread(url)
    results = model_detection(pic)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get class ID: 0, 1, 2
        cls_id = int(box.cls[0]) 
        # Get class name
        cls_name = translate_label(cls_id)
        

        # Optional: Get confidence score
        conf = float(box.conf[0])
        cv2.rectangle(pic, (x1, y1), (x2, y2), (0,255,0), 2)
        cropped_pic = pic[y1 -MARGIN: y2 +MARGIN, x1 -MARGIN: x2 +MARGIN]
        # Draw label
        conf = round(conf*100., 2)
        # label = f"{cls_name}"
        # cv2.putText(pic, label, (x1, y1 - 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    resized_pic = cv2.resize(cropped_pic, (500, 400))
    
    return resized_pic, cropped_pic, cls_name

    # cv2.imshow("Crop raw", cropped_pic)
    # cv2.imshow("Container Prediction", resized_pic)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def real_time_video(url, iterval = 0.1):
    model_classification, class_to_idx = load_model_classification()

    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_order = 0

    if not cap.isOpened():
        print("Not play video")
        return None
    
    while cap.isOpened():
        last_time = time.perf_counter()
        ret, pic = cap.read()
        if not ret: break

        if frame_order % SKIP_FRAME != 0:
            frame_order += 1
            continue
            
    ### =========== BEGIN PROCESSING FRAME ==============
        # ======   CONFIGURE    ===========
        print(f"\n\n==============\n Frame {frame_order}")
        h, w = pic.shape[:2]
        w_begin = int(w/4)
        w_end = int(w* 0.9) + MARGIN
        print(f"line {w_begin}-{w_end}")
        display_pic = pic.copy()
        cv2.line(display_pic, (w_begin, 0), (w_begin, h), (0,255,255),2)
        cv2.line(display_pic, (w_end, 0), (w_end, h), (0,255,255),2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = .7
        thickness = 2

        # ======   LOAD DETECTION MODEL    ===========
        results = model_detection(pic)
        confidiences = results[0].boxes.conf

        # ====== FOUND THE BBOX WITH MAX CONFIDIENCE ===========
        max_idx = 0
        num_confidiences = len(confidiences)
        print(f"num_confd: {num_confidiences}")
        if num_confidiences == 0: continue

        for idx in range (num_confidiences):
            print(f"    idx: {idx} - {confidiences[idx]}")
            if confidiences[idx] > confidiences[max_idx] and confidiences[idx] > CONF_DETECT_THRESH:
                max_idx = idx
        max_bbox = results[0].boxes[max_idx]
        x1, y1, x2, y2 = map(int, max_bbox.xyxy[0])
        print(f"x1, y1, x2, y2 {x1, y1, x2, y2}")
        # =================================
        
        # Get class_name for ROI container
        cls_id = int(max_bbox.cls[0])
        label_container = translate_label(cls_id)
        conf_detection = float(max_bbox.conf[0])

        # Draw bounding box
        cv2.rectangle(display_pic, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1 = max(0, x1 - MARGIN)
        y1 = max(0, y1 - MARGIN)
        x2 = min(w, x2 + MARGIN)
        y2 = min(h, y2 + MARGIN)
        cropped_pic = pic[y1: y2, x1: x2]
        if cropped_pic.size == 0: 
            print("Warning: Empty crop detected!")
            continue            

        # ===== PROCESS IN CASE OF EMPTY CONTAINER ==========
        if label_container == "het hang":
            cv2.putText(display_pic, label_container, (x1, y1), fontFace=font, fontScale=font_scale, color=(0, 255, 0), thickness=thickness) 
            continue

        # ===== PROCESS IN CASE OF NOT EMPTY CONTAINER ==========
        # Requirement:
        # 1. If edges's container over line w_begin and w_end
        # Beginning Image Classification
        # 2. After classification, hold 6-10 frames to get certain result (label_score)
        #
        #
        if x1 >= w_begin and x2 >= w_end : 
            print(f"w4/x1: {w_begin}/{x1}")

            # Image classification
            print(f"  ======== BEGIN CLASSIFICATION =========")
            pil_img = Image.fromarray(cv2.cvtColor(cropped_pic, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess_image(pil_img)
            # Inference
            with torch.no_grad():
                outputs = model_classification(input_tensor)
                probability = nn.functional.softmax(outputs[0], dim=0)
                best_prob, best_class = torch.max(probability, 0)

            # Get class name and confidence
            goods_name = get_class_name(best_class.item(), class_to_idx)
            conf_goods = round(best_prob.item()*100., 2)
            label_goods = f"{goods_name}: {conf_goods}%" if label_container != "het hang" else ""
            print(f"label_classification: {label_goods}")

            # Display prediction on frame
            (label_w,_), _ = cv2.getTextSize(label_goods, font, font_scale, thickness)
            cv2.putText(display_pic, label_container, (x1, y1), fontFace=font, fontScale=font_scale, color=(0, 255, 0), thickness=thickness)
            cv2.putText(display_pic, label_goods, (x2 - label_w, y1), fontFace=font, fontScale=font_scale, color=(0, 255, 0), thickness=thickness)
            print(f"  ======== END CLASSIFICATION =========")
        # ===========================
        
        # Display the resulting frame
        resized_pic = cv2.resize(display_pic, (600, 400))
        cv2.imshow('Classify goods', resized_pic)

        ## After for loop
        process_time = time.perf_counter() - last_time
        remain_time = max(0, iterval - process_time)
        print(f"pro_time: {process_time}")
        print(f"remain_time: {remain_time}")

        if remain_time > 0:
            key = cv2.waitKey(int(remain_time*1000))
        else:
            key = cv2.waitKey(1)
        
        if key == ord('q'): break
        frame_order += 1
    ### ========== END PROCESSING FRAME =============
    cap.release()
    cv2.destroyAllWindows()


def detect_container_video(url):
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_counter = 0
    while cap.isOpened():
        ret, pic = cap.read()
        if not ret: break
        frame_counter += 1
        results = model_detection(pic)
        masks = results[0].boxes.conf > CONF_DETECT_THRESH
        filtered_conf = results[0].boxes[masks]
        for i, box in enumerate(filtered_conf):
            print(f"    ======\n  Box conf {i}. of frame {frame_counter}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"x1, y1, x2, y2 {x1, y1, x2, y2}")
            # Get class ID
            cls_id = int(box.cls[0])

            # Get class name
            cls_name = translate_label(cls_id)

            # Optional: Get confidence score
            conf = float(box.conf[0])

            # Draw bounding box
            cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cropped_pic = pic[y1 -MARGIN: y2 +MARGIN, x1 -MARGIN: x2 +MARGIN]
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(pic, label, (x1, y1 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        resized_pic = cv2.resize(pic, (500, 400))
        cv2.imshow("Container Detection", resized_pic)
        if cv2.waitKey(25) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    # return resized_pic, cropped_pic, cls_name

if __name__ == "__main__":
    real_time_video("rtsp://admin:longson2016@192.168.10.26:1555/Streaming/Channels/5002")
    # real_time_video("./data/video_03.mp4")
    
    # detect_container_video("./data/video_02.mp4")