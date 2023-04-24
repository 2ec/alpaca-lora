from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import sys
sys.path.insert(0, "/Users/emilchri/Documents/Robotikk/Master/Oppgave/Kode/alpaca/alpaca-lora/")
from alpaca_image_feature_extraction_torch import get_image_top_n_classes_faster_rcnn
import sys
import pandas as pd
import numpy as np

data_path = "/Users/emilchri/Documents/Robotikk/Master/Oppgave/Kode/alpaca/alpaca-lora/ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/med_qa_imageid_without_not_relevant_20000.json"
df = pd.read_json(data_path)

DATA_PATH = "med_qa_imageid_5000.json"
IMAGE_PATH = "/Users/emilchri/Documents/Robotikk/Master/Oppgave/Kode/alpaca/alpaca-lora/ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/images"
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
IMAGE_MODEL = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.001)
IMAGE_MODEL.eval()
TOP_N_IMAGE_FEATURES = 50



def get_img_features(data_point):
    img_path = f"{IMAGE_PATH}/{data_point['input']}.jpg"


    img_features = (
        get_image_top_n_classes_faster_rcnn(
            img=img_path,
            model=IMAGE_MODEL,
            top_n_features=TOP_N_IMAGE_FEATURES,
            from_path=True,
        )
    )
    return img_features


prev_input = ""
prev_img_features = []
img_features_results = np.arange(20000, dtype=object)
counter = 0
for index, row in df.iterrows():
    
    # print(row['instruction'], row['input'])
    if row['input'] == prev_input:
        img_features = prev_img_features
    else:
        img_features = get_img_features(row)
        prev_input = row['input']
        prev_img_features = img_features
    # print(img_features)
    img_features_results[index] = img_features
    counter += 1

    if counter % 200 == 0:
        print(counter)
        


df["input_image"] = img_features_results
df.to_json("/Users/emilchri/Documents/Robotikk/Master/Oppgave/Kode/alpaca/alpaca-lora/ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/med_qa_imageid_without_not_relevant_20000_with_features.json")