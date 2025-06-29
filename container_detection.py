from ultralytics import YOLO
import cv2, os
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image   
import time
from setting import MODEL_DETECTION, CONF_DETECT_THRESH, SKIP_FRAME
from goods_classification import load_model_classification, get_class_name, preprocess_image

MARGIN = 50
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

    # cv2.namedWindow('Phan loai hang hoa', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Phan loai hang hoa', 600, 400)

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
            
        # Processing a frame
        print(f"\n\n==============\n Frame {frame_order}")
        h, w = pic.shape[:2]
        display_pic = pic.copy()

        results = model_detection(pic)
        masks = results[0].boxes.conf > CONF_DETECT_THRESH
        filtered_boxes = results[0].boxes[masks]

        for i, box in enumerate(filtered_boxes):
            print(f"    ======\n  Box conf {i}. of frame {frame_order}")
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"x1, y1, x2, y2 {x1, y1, x2, y2}")
            
            # Get class_name for ROI container
            cls_id = int(box.cls[0])
            label_container = translate_label(cls_id)
            conf_detection = float(box.conf[0])

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
            
            # Image classification
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
            label_goods = f"{goods_name}: {conf_goods}%"
            print(f"label_classification: {label_goods}")

            # Display prediction on frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 2
            margin = 20
            (label_w, label_h), _ = cv2.getTextSize(label_goods, font, font_scale, thickness)
            x = w - label_w - margin
            y = label_h + margin
            cv2.putText(display_pic, label_container, (margin, y), fontFace=font, fontScale=font_scale, color=(0, 255, 0), thickness=thickness)
            cv2.putText(display_pic, label_goods, (x, y), fontFace=font, fontScale=font_scale, color=(0, 255, 0), thickness=thickness)

        # Display the resulting frame
        resized_pic = cv2.resize(display_pic, (600, 400))
        cv2.imshow('Classify goods', resized_pic)

        ## After for loop
        process_time = time.perf_counter() - last_time
        remain_time = max(0, iterval - process_time)
        if remain_time > 0:
            key = cv2.waitKey(int(remain_time*1000))
        else:
            key = cv2.waitKey(1)
        
        if key == ord('q'): break
        frame_order += 1

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
    # real_time_video("rtsp://admin:longson2016@192.168.10.26:1555/Streaming/Channels/5002")
    real_time_video("./data/video_01.mp4")
    # detect_container_video("./data/video_01.mp4")

    # url_input = detect_container_img("./data/img01.jpg")
