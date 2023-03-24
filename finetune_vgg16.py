from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
import sys

import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights
import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import alpaca_image_feature_extraction_torch
import numpy as np

#from tensorflow.keras.applications.vgg16 import VGG16


assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"


# optimized for RTX 4090. for larger GPUs, increase some of these?
MICRO_BATCH_SIZE = 3  # this could actually be 5 but i like powers of 2
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  # we don't always need 3 tbh
LEARNING_RATE = 3e-4  # the Karpathy constant
CUTOFF_LEN = 1485 #256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 11000 # 30% of the dataset  # 2000
TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
# IMAGE_MODEL = VGG16(weights="imagenet", include_top=False)
weights = VGG16_Weights.IMAGENET1K_V1
IMAGE_MODEL = vgg16(weights=weights)
IMAGE_MODEL.eval()
#IMAGE_MODEL = VGG16(weights="imagenet", include_top=True, classes=1000, pooling=None)
TOP_N_IMAGE_FEATURES = 100
# "ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/med_vqa_imageid.json"
DATA_PATH = "med_qa_imageid.json"
IMAGE_PATH = "ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/images"
OUTPUT_DIR = "lora-alpaca"
GLOBAL_LAST_PROMPT = {"ImageID": "", "image_features": ""}

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    device_map=device_map,
)
tokenizer = LlamaTokenizer.from_pretrained(
    "decapoda-research/llama-7b-hf", add_eos_token=True
)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
data = load_dataset("json", data_files=DATA_PATH)


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    # if GLOBAL_LAST_PROMPT["ImageID"] == data_point["input"]: # Checks if the image is the same as the last and don't compute new features if already done.
    #     vgg16_img_features = GLOBAL_LAST_PROMPT["image_features"]
    # else:
    img_path = f"{IMAGE_PATH}/{data_point['input']}.jpg"
    vgg16_img_features = alpaca_image_feature_extraction_torch.get_image_top_n_classes(img_path=img_path, model=IMAGE_MODEL, top_n_features=TOP_N_IMAGE_FEATURES)
    # GLOBAL_LAST_PROMPT["ImageID"] = data_point['input']
    # GLOBAL_LAST_PROMPT["image_features"] = vgg16_img_features

    return f"""Below is an question that describes a task, paired with image featues that descirbes the top {TOP_N_IMAGE_FEATURES} image classes and their score. Write a response that appropriately answers the question usning the image information.

### Instruction:
{data_point["instruction"]}

### Image features:
{vgg16_img_features}

### Response:
{data_point["output"]}"""


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    prompt = generate_prompt(data_point)
    return tokenize(prompt)


if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=60 if VAL_SET_SIZE > 0 else None,
        save_steps=60,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

trainer.train()
print("\nFinished training!")
model.save_pretrained(OUTPUT_DIR)
print(f"\nSaved model to {OUTPUT_DIR}")
print("\n If there's a warning about missing keys above, please disregard :)")