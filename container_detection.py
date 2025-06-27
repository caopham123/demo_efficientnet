from ultralytics import YOLO
import cv2, os
from setting import MODEL_DETECTION

model_file = os.path.join(MODEL_DETECTION, "best.py")
model = YOLO()
class_names = model.names


def translate_label(idx_class):
    vn_lst_name = ["het hang", "con hang"]

    return vn_lst_name[idx_class]


def detect_container_img(url):
    pic = cv2.imread(url)
    results = model(pic)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Get class ID: 0, 1, 2
        cls_id = int(box.cls[0]) 
        # Get class name
        cls_name = translate_label(cls_id)
        

        # Optional: Get confidence score
        conf = float(box.conf[0])
        cv2.rectangle(pic, (x1, y1), (x2, y2), (0,255,0), 2)
        # Draw label
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(pic, label, (x1, y1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    resized_pic = cv2.resize(pic, (800, 600))
    cv2.imshow("Container Prediction", resized_pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_container_video(url):
    cap = cv2.VideoCapture(url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Get class ID
            cls_id = int(box.cls[0])

            # Get class name
            cls_name = translate_label(cls_id)

            # Optional: Get confidence score
            conf = float(box.conf[0])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 4)

        resized_frame = cv2.resize(frame, (800, 800))
        cv2.imshow("Container Detection", resized_frame)
        if cv2.waitKey(1) == 27: break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url_input = detect_container_img("img_03.jpg")
    # url_input = pred_video("video_03.mp4")