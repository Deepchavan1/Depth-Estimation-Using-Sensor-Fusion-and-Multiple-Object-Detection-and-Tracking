import torch
import numpy as np
from super_gradients.training import models
from mot import yolonas_deepSORT
import os
import cv2


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
 
model = models.get("yolo_nas_s", pretrained_weights="coco").to(device)


custom_tracker = yolonas_deepSORT("/home/deep/yolo-nas/deep_sort/model_weights/mars-small128.pb", model)

frame_number = 0
folder_path = "./inputs/"
for img in os.listdir(folder_path):
    print(folder_path + img)
    frame = cv2.imread(folder_path + img)
    frame_number += 1
    custom_tracker.track(frame, "./outputs", frame_number, save_outputs=True)