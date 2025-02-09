import os
import sys
from typing import List

import fire
import torch
import transformers
from datasets import load_dataset
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

import alpaca_image_feature_extraction_torch

# assert (
#     "LlamaTokenizer" in transformers._import_structure["models.llama"]
# ), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer


DATA_PATH = "med_qa_imageid_5000.json"
IMAGE_PATH = "ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/images"
MAP_NUM_PROC = 1 #os.cpu_count() # Seems to not work on compute cluster per now.
global TOP_N_IMAGE_FEATURES
global IMAGE_MODEL

def train(
    # model/data params
    base_model: str = "decapoda-research/llama-7b-hf",  # the only required argument
    data_path: str = DATA_PATH,
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 3,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1588,
    val_set_size: int = 6000,  # 30% of train set
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve,
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    image_feature_extractor: str = "faster_rcnn"
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
        f"image_feature_extractor: {image_feature_extractor}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    # Choose image feature extraction method
    implemented_image_feature_extractions = ["vgg16", "faster_rcnn"]
    assert (image_feature_extractor in implemented_image_feature_extractions), f"Please choose one of the models: {implemented_image_feature_extractions}."

    global IMAGE_MODEL
    global TOP_N_IMAGE_FEATURES
    global TOP_N_IMAGE_FEATURES
    if image_feature_extractor == "vgg16":
        print("\nLoading VGG16 pretrained on ImageNet and 100 top image features.")
        weights = VGG16_Weights.IMAGENET1K_V1
        IMAGE_MODEL = vgg16(weights=weights)
        IMAGE_MODEL.eval()
        TOP_N_IMAGE_FEATURES = 100
        image_feature_extractor_func = alpaca_image_feature_extraction_torch.get_image_top_n_classes
        img_encoder_structure = "(label, probability)"
    elif image_feature_extractor == "faster_rcnn":
        print("\nLoading Faster RCNN pretrained on COCO and 50 top image features.")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        IMAGE_MODEL = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.001)
        IMAGE_MODEL.eval()
        TOP_N_IMAGE_FEATURES = 50
        
        image_feature_extractor_func =  alpaca_image_feature_extraction_torch.get_image_top_n_classes_faster_rcnn
        img_encoder_structure = "(label, probability, coordinates(xmin,ymin,xmax,ymax))"
    else:
        raise NotImplementedError(f"Could not find the model {image_feature_extractor} in {implemented_image_feature_extractions}")

    if torch.__version__ >= "2" and sys.platform != "win32":
        IMAGE_MODEL = torch.compile(IMAGE_MODEL)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map=device_map,
        torch_dtype=torch.float16,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result


    def generate_prompt(data_point, image_feature_extractor_func, img_encoder_structure):
        global IMAGE_PATH
        global IMAGE_MODEL
        global TOP_N_IMAGE_FEATURES
        # sorry about the formatting disaster gotta move fast
        img_path = f"{IMAGE_PATH}/{data_point['input']}.jpg"
        
        img_features = (
            image_feature_extractor_func(
                img=img_path,
                model=IMAGE_MODEL,
                top_n_features=TOP_N_IMAGE_FEATURES,
                from_path=True,
            )
        )
        return f"""Below is a question that describes a task, paired with an input that provides image features from an encoded image. The form of the image features are {img_encoder_structure}. Write a response that appropriately completes the request.

### Question:
{data_point["instruction"]}

### Encoded image features on the form {img_encoder_structure}:
{img_features}

### Answer:
{data_point["output"]}"""


    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point, image_feature_extractor_func, img_encoder_structure)
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    data = load_dataset("json", data_files=data_path)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"]
            .shuffle()
            .map(generate_and_tokenize_prompt, num_proc=MAP_NUM_PROC)
        )
        val_data = (
            train_val["test"]
            .shuffle()
            .map(generate_and_tokenize_prompt, num_proc=MAP_NUM_PROC)
        )
    else:
        train_data = (
            data["train"]
            .shuffle()
            .map(generate_and_tokenize_prompt, num_proc=MAP_NUM_PROC)
        )
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            optim="adamw_torch",
            eval_steps=20 if val_set_size > 0 else None,
            save_steps=20,
            do_eval=True,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            torch_compile=True
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
