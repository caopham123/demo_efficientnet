import cv2, os
import torch, timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np   
from ai_core.src.setting import *
from ai_core.src.container_detection import ContainerDetection

class GoodsClassification:

    def __init__(self):
        self.detection = ContainerDetection()

    # ----- CREATE MODEL & LOAD WEIGHTS ------
    def load_classification_model(self):
        self.checkpoint = torch.load(os.path.join(MODEL_CLASSIFICATION,'best.pt').replace('\\','/')
                                , map_location=DEVICE)
        print(f"cls_to_idx: {self.checkpoint['class_to_idx']}")
        self.num_classes = len(self.checkpoint['class_to_idx'])

        self.model = timm.create_model(
                model_name=MODEL_NAME, pretrained=False
                ,num_classes = self.num_classes)
        self.model.to(DEVICE)

        self.model.load_state_dict(self.checkpoint['state_dict'])
        self.model.eval()
        return self.model, self.checkpoint['class_to_idx']
    # --------------------------------

    def preprocess_image(self, image):
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return self.transform(image).unsqueeze(0).to(DEVICE)

    def get_class_name(self, class_idx, class_to_idx: dict):
        # Convert class key to index key
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}
        class_name = idx_to_class.get(int(class_idx), "Ko_xac_dinh")
        class_name = str(class_name).replace('_', ' ')
        return class_name

    def classify_goods_img(self, img_input:str):
        if self.detection.detect_container_by_path(img_input) is None:
            print("Detection is None")
            return None
        
        model, class_to_idx = self.load_classification_model()
        display_pic, cropped_pic_lst, container_label_lst, container_score_lst = self.detection.detect_container_by_path(img_input)
        number_boxes= len(cropped_pic_lst)

        goods_label_lst= []
        goods_score_lst= []
        for i in range(number_boxes):
            if container_label_lst[i] == "het hang":
                goods_label_lst.append("het hang")
                goods_score_lst.append(container_score_lst[i])
                
            pil_img = Image.fromarray(cv2.cvtColor(cropped_pic_lst[i], cv2.COLOR_BGR2RGB))
            input_tensor = self.preprocess_image(pil_img)

            # Inference goods image
            with torch.no_grad():
                outputs = model(input_tensor)
                probability = nn.functional.softmax(outputs[0], dim=0)
                best_prob, best_class = torch.max(probability, 0)
            # print(f"Best prob: {best_prob} ==== Match class idx: {best_class}")

            # Get class name and confidence
            class_name = self.get_class_name(best_class.item(), class_to_idx)
            confidence = best_prob.item()*100.
            confidence = round(confidence, 2)
            goods_label_lst.append(class_name)
            goods_score_lst.append(confidence)

        return display_pic, goods_label_lst, goods_score_lst

    # def show_classified_img(self, img_input):
    #     display_pic, goods_label, container_label = self.classify_goods_img(img_input)
    #     if display_pic is None:
    #         print("Not classify goods")
    #         return None
    #     # Display prediction on frame
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = .6
    #     thickness = 2
    #     margin = 10

    #     combined_label = f"{container_label} | {goods_label}" if container_label != "het hang" else container_label

    #     # print(f"label: {goods_label}")
    #     display_pic= cv2.putText(display_pic, combined_label, (margin, margin*2), fontFace=font
    #                               ,fontScale=font_scale, color=(0, 255, 0), thickness=thickness)
        
    #     cv2.imshow('Classify goods', display_pic)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     return display_pic

if __name__ == "__main__":
    goods_classification = GoodsClassification()
    # classify_goods_video("./data/video_01.mp4")
    goods_classification.show_classified_img('./data/img02.jpg')
    print("Done")