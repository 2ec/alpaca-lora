from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

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

def get_image_top_n_classes(img_path:str, model, top_n_features:int=100) -> list:
    print("\nImage path:", img_path)
    x = preprocess_img(img_path)
    print(type(x))
    yhat = model.predict(x)
    labels = decode_predictions(yhat, top=top_n_features)
    labels = labels[0][:]
    labels = [(label.replace('"', "'"), f"{score*100:.3f}") for id, label, score in labels] # Clean up: replace quotes and return percentages with 3 decimals.
    return labels

def feature_to_json(features: np.ndarray, save_path: str):
    features_list = features.tolist()

    with open(save_path, "w") as f:
        json.dump(features_list, f)


if __name__ == "__main__":
    image_path = "ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/images/cl8k2u1pm1dw7083203g1b7yv.jpg"
    print(f"\nImage path: {image_path} \nLoading model...")
    model = VGG16(weights="imagenet", include_top=True, classes=1000, pooling=None)
    labels = get_image_top_n_classes(img_path=image_path, model=model)

    print(f"\nTop 100 labels for iamge: \n{labels}")
