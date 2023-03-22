import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import json


def preprocess_img(img_path: str) -> np.ndarray:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def get_image_features(img_path: str, model, np_type: np.dtype = np.float16) -> np.ndarray:
    x = preprocess_img(img_path)
    features = model.predict(x)[0].astype(np_type)
    return features


def feature_to_json(features: np.ndarray, save_path: str):
    features_list = features.tolist()

    with open(save_path, "w") as f:
        json.dump(features_list, f)
