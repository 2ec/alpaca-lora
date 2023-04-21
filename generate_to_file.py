import sys
import torch
import torch.nn.functional as F
from peft import PeftModel
import transformers
import json
from torchvision.models import VGG16_Weights, vgg16

from alpaca_image_feature_extraction_torch import get_image_top_n_classes

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

LOAD_8BIT = False
BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = input("\nPress enter for default weights or enter path: ")


weights = VGG16_Weights.IMAGENET1K_V1
IMAGE_MODEL = vgg16(weights=weights)
IMAGE_MODEL.eval()
TOP_N_IMAGE_FEATURES = 100
DATA_PATH = "ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/med_qa_imageid_without_not_relevant_5000_test.json"
IMAGE_PATH = "ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset/images"
NEW_ANSWERED_FILE_PATH = "results/med_qa_imageid_5000_test_answered.json"

if not LORA_WEIGHTS:
    print("Loading default weights...")
    LORA_WEIGHTS = "tloen/alpaca-lora-7b"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        torch_dtype=torch.float16,
    )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    model = PeftModel.from_pretrained(
        model,
        LORA_WEIGHTS,
        device_map={"": device},
    )


def generate_prompt(data_point, image_feature_extractor_func=get_image_top_n_classes, img_encoder_structure="(label, probability)"):
    # sorry about the formatting disaster gotta move fast
    instruction, input = data_point["instruction"], data_point["input"]
    
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

### Response:""", img_features, data_point["output"]

if not LOAD_8BIT:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)


def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=256,
    **kwargs,
):
    prompt, img_features, correct_answer = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    scores = generation_output.scores[0].softmax(1).detach().cpu().numpy()
    return output, scores, img_features, correct_answer


def main():
    def append_result(content_dict, file_path):
        with open(file_path, "a") as f:
            fs = str(content_dict).replace("'",'"')
            f.write(f"\t{fs},\n")
    

    with open(DATA_PATH, "r") as f:
        content = f.read()

    parsed = json.loads(content)

    for question_answer in parsed:
        output, scores, img_features, answer = evaluate(question_answer)
        output_cleaned = output.split('### Response:')[1].strip()
        question_answer["output_answered"] = output_cleaned
        question_answer["input"] = img_features
        append_result(question_answer, NEW_ANSWERED_FILE_PATH)

        

if __name__ == "__main__":
    main()
    # adapter_config.json
    # '/lora-alpaca/20_5/adapter_config.json
    # lora-alpaca/20_5/adapter_config.json