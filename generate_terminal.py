import sys
import torch
from peft import PeftModel
import transformers
import numpy as np
import lime
import lime.lime_text

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")

LOAD_8BIT = False
BASE_MODEL = "decapoda-research/llama-7b-hf"
LORA_WEIGHTS = input("\nPress enter for default weights or enter path: ")


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


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

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
    prompt = generate_prompt(instruction, input)
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
    scores = generation_output[0].softmax(1).detach().numpy()
    return output, scores

def alpaca_predict_lime(texts):
    return np.array([evaluate(instruction, input_token) for text in texts])

# Set up LIME explainer
EXPLAINER = lime.lime_text.LimeTextExplainer(verbose=True)

while True:
    instruction=input("\nEnter instruction. Press enter to exit. ")
    if len(instruction) <= 0:
        print("Breaking free!")
        break
    input_token = input("Enter optional input: ")
    output, scores = evaluate(instruction, input_token)
    output_cleaned = output.split('### Response:')[1].strip()

    print(f"\nResponse: {output_cleaned}")
    
    see_more = input("Do you want to see the whole output? y/n: ")
    if see_more == "y":
        print(f"\nWhole output string:\n{output}\n\ns:\n{s}")

    want_lime = input("Do you want LIME? y/n: ")
    if want_lime == "y":
        num_features = input("How many features do you want in the explenation? Default is 10. ")
        # Explain predictions using LIME
        exp = EXPLAINER.explain_instance(instruction, scores, num_features=num_features)
        #exp.show_in_notebook()

        
        
        file_path = input("Input file path to save image. End with .html ")

  
        exp.save_to_file(file_path)
