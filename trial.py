import torch
import numpy as np
from super_gradients.training import models
import cv2

img = cv2.imread("./inputs/horses.jpg")
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
 
model = models.get("yolo_nas_s", pretrained_weights="coco").to(device)

out = model.predict(img, conf=0.6)
preds = out._images_prediction_lst

print(preds)

for pred in preds:
    yolo_dets = pred.prediction
    print(yolo_dets.bboxes_xyxy)
    print(yolo_dets.confidence)
    print(yolo_dets.labels)
    print(yolo_dets.bboxes_xyxy.shape[0])

# out.save("outputs")
# print()
# print("----------------------------------------------------------------")
# print(out)
# out_np = np.array(out)
# print()
# print("----------------------------------------------------------------")
# print(out_np)