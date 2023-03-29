from torchvision.io import read_image
from torchvision.models import vgg16, VGG16_Weights

def get_image_top_n_classes(img, model, top_n_features:int=100, weights=VGG16_Weights.IMAGENET1K_V1, from_path=True) -> list:
    predicted = get_image_predictions(img, model, weights, from_path)
    predicted_softmax = predicted.softmax(0)

    new_list = []
    for i in range(len(predicted_softmax)):
        score = round(predicted_softmax[i].item()*100, 3)
        category_name = weights.meta["categories"][i]
        new_list.append((category_name, score))

    new_list_sorted = sorted(new_list, key=lambda tup: tup[1], reverse=True)
    return new_list_sorted[:top_n_features]

def get_image_predictions(img, model, weights=VGG16_Weights.IMAGENET1K_V1, from_path=True):
    if from_path:
        img = read_image(img)

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    batch = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    predictions = model(batch).squeeze(0)
    return predictions


if __name__ == "__main__":
    # From https://pytorch.org/vision/stable/models.html#classification
    # Step 1: Initialize model with the best available weights
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights)
    model.eval()
    NUM_TOP_CATEGORIES = 100

    img = "ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/images/cl8k2u1pm1dw7083203g1b7yv.jpg"

    # Step 4: Use the model and print the predicted category
    predictions = get_image_predictions(img_path=img, model=model)

    prediction = predictions.softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"Top category: {category_name}: {100 * score:.1f}%")

    top_n_classes = get_image_top_n_classes(img_path=img, model=model, top_n_features=NUM_TOP_CATEGORIES)

    print(top_n_classes)

