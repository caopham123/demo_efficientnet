import cv2, os
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image   
from main2 import create_model
from setting import *


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# ----- CREATE MODEL & LOAD WEIGHTS ------
def load_model():
    checkpoint = torch.load(os.path.join(MODEL_DIR,'best.pt').replace('\\','/')
                            , map_location=DEVICE)
    print(f"cls_to_idx: {checkpoint['class_to_idx']}")
    num_classes = len(checkpoint['class_to_idx'])

    model = timm.create_model(
            model_name=MODEL_NAME, pretrained=False
            , num_classes = num_classes)
    model.to(DEVICE)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint['class_to_idx']
# --------------------------------


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def get_class_name(class_idx, class_to_idx):
    # Convert class key to index key
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    print(f"idx_to_class: {idx_to_class}")
    class_name = idx_to_class.get(int(class_idx), "Ko_xac_dinh")
    class_name = class_name.replace('_', ' ')
    print(f"Result: {class_name}")
    
    return class_name

def classify_goods_video(video_url):
    model, class_to_idx = load_model()

    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise IOError("Not found video")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        print(h, w)

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_tensor = preprocess_image(pil_img)

        # Inference image
        with torch.no_grad():
            outputs = model(input_tensor)
            probability = nn.functional.softmax(outputs[0], dim=0)
            best_prob, best_class = torch.max(probability, 0)
        print(f"Best prob:{best_prob*100:2%}\nMatch class idx:{best_class}")

        # Get class name and confidence
        class_name = get_class_name(best_class.item(), class_to_idx)
        confidence = best_prob.item()
        # Display prediction on frame
        label = f"{class_name}: {confidence*100.:0%}%"
        cv2.putText(frame, label, (w, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Display the resulting frame
        resized_frame = cv2.resize(frame, (800, 600))
        cv2.imshow('Webcam Classification', resized_frame)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def classify_goods_img(img_path):
    model, class_to_idx = load_model()
    pic = cv2.imread(img_path)
    # if not pic:
    #     raise FileNotFoundError(f"Not found path {img_path}")
    h, w = pic.shape[:2]
    h3, w3 = h-300, w-400

    pil_img = Image.fromarray(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess_image(pil_img)

    # Inference image
    with torch.no_grad():
        outputs = model(input_tensor)
        probability = nn.functional.softmax(outputs[0], dim=0)
        best_prob, best_class = torch.max(probability, 0)
    
    print(f"Best prob: {best_prob}\nMatch class idx: {best_class}")

    # Get class name and confidence
    class_name = get_class_name(best_class.item(), class_to_idx)
    confidence = best_prob.item()*100.
    # Display prediction on frame
    label = f"{class_name}: {confidence}%"
    print(f"label: {label}")
    cv2.putText(pic, label, (w3, h3), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    # Display the resulting frame
    resized_frame = cv2.resize(pic, (800, 600))
    cv2.imshow('Webcam Classification', resized_frame)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    # classify_goods_video("video_01.mp4")
    classify_goods_img('img01.jpg')
    # class_to_idx = {'class01': 0, 'class02': 1}
    # get_class_name(10, class_to_idx)
    # get_class_name(1, class_to_idx)
    print("Done")