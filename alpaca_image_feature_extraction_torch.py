from torchvision.io import read_image
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import pandas as pd


def get_image_top_n_classes(
    img,
    model,
    top_n_features: int = 100,
    weights=VGG16_Weights.IMAGENET1K_V1,
    from_path=True,
) -> list:
    predicted = get_image_predictions(img, model, weights, from_path)
    predicted_softmax = predicted.softmax(0)

    new_list = []
    for i in range(len(predicted_softmax)):
        score = round(predicted_softmax[i].item() * 100, 3)
        category_name = weights.meta["categories"][i]
        new_list.append((category_name, score))

    new_list_sorted = sorted(new_list, key=lambda tup: tup[1], reverse=True)
    return new_list_sorted[:top_n_features]


def get_image_predictions(
    img, model, weights=VGG16_Weights.IMAGENET1K_V1, from_path=True
):
    if from_path:
        img = read_image(img)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms(antialias=True)

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    predictions = model(batch).squeeze(0)
    return predictions


def get_image_predictions_faster_rcnn(img, model, weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, from_path=True):
    if from_path:
        img = read_image(img)

        # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and visualize the prediction
    prediction = model(batch)[0]
    return prediction

def get_image_top_n_classes_faster_rcnn(
        img,
        model,
        top_n_features: int = 50,
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        from_path=True,
    ) -> list:
    prediction = get_image_predictions_faster_rcnn(img, model, weights, from_path)

    labels_scores = []

    for i, score in enumerate(prediction["scores"]):
        label = weights.meta["categories"][prediction["labels"][i]]
        xmin, ymin, xmax, ymax = prediction["boxes"][i].detach().tolist()
        coordinates = (int(xmin), int(ymin), int(xmax), int(ymax))
        score = round(score.item() * 100, 3)
        labels_scores.append([label, score, coordinates])
    sorted_labels_scores = sort_list_of_scores(labels_scores, top_n_features)
    return sorted_labels_scores

def sort_list_of_scores(list:list, top_n_features:int) -> list:
    df = pd.DataFrame(list, columns=["label", "score", "coordinates"])
    df.sort_values(by=["score"], ascending=False, inplace=True)
    return df.loc[:top_n_features].values.tolist()


if __name__ == "__main__":
    # From https://pytorch.org/vision/stable/models.html#classification
    # Step 1: Initialize model with the best available weights
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.001)

    #weights = VGG16_Weights.IMAGENET1K_V1
    #model = vgg16(weights=weights)
    model.eval()
    NUM_TOP_CATEGORIES = 100

    img = "ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/images/cl8k2u1pm1dw7083203g1b7yv.jpg"

    # Step 4: Use the model and print the predicted category
    sorted_labels_scores = get_image_top_n_classes_faster_rcnn(img=img, model=model, top_n_features=50, weights=weights, from_path=True)
    for i in sorted_labels_scores:
        print(i)