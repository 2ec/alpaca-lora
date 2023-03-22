import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import json

model = VGG16(weights="imagenet", include_top=False)

!pip install gdown
!gdown 1jTyLWwcHzbLpWjSNwmgiiavXDjuQe5y7 - O / content/
!jar xvf / content/ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset.zip

ROOT_PATH = "/content/ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset"
IMG_PATH = f"{ROOT_PATH}/images/"

image_list = os.listdir(IMG_PATH)
print(f"{IMG_PATH}{image_list[0]}")
print(len(image_list))


def preprocess_img(img_path: str) -> np.ndarray:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def get_features_from_image(img_path: str, np_type=np.float16) -> np.ndarray:
    x = preprocess_img(img_path)
    features = model.predict(x)[0].astype(np_type)
    return features


def feature_to_json(features: np.ndarray, save_path: str):
    features_list = features.tolist()

    with open(save_path, "w") as f:
        json.dump(features_list, f)


#img_path = IMG_PATH + "/" + "cla820glss4vz071ua1jw6jco" + ".jpg"
img_path = "/content/clb0lbwzldpbo086u2vfkeh30.jpg"
x = preprocess_img(img_path)

features = model.predict(x)[0].astype(np.float16)
print(features.shape)
print(type(features))
print(features.dtype)

feature_to_json(features, f"/content/clb0lbwzldpbo086u2vfkeh30.json")

!rm - r / content/ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/json / & & mkdir / content/ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/json/

IMG_EXTENSION = ".jpg"

for img in image_list:
    img_name = img.split(IMG_EXTENSION)[0]
    img_path = f"{IMG_PATH}/{img}"
    save_path = f"{ROOT_PATH}/json/{img_name}.json"

    x = preprocess_img(img_path)
    features = model.predict(x)[0]

    feature_to_json(features=features, save_path=save_path)

!zip - r / content/VGG16_features_json.zip / content/ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/json
