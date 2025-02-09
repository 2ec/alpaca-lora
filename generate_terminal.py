import sys
import torch
import torch.nn.functional as F
from peft import PeftModel
import transformers
import numpy as np
import lime
from lime.lime_text import LimeTextExplainer
#from eli5.lime import TextExplainer
from typing import List

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
    scores = generation_output.scores[0].softmax(1).detach().cpu().numpy()
    return output, scores, generation_output

def alpaca_predict_lime(texts):
    scores = []
    for text in texts:
        output, score, generation_output = evaluate(text)
        
    
    return np.array([evaluate(instruction, input_token) for text in texts])

def model_adapter(texts: List[str]) -> np.ndarray:
    all_scores = []

    for i in range(0, len(texts), 64):
        text_batch = texts[i:i+64]
        # use Llama encoder to tokenize text 
        encoded_input = tokenizer(text_batch, return_tensors="pt")
        # run the model
        input_ids = encoded_input["input_ids"].to(device)
       
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
        #scores = output.scores[0].softmax(1).detach().cpu().numpy()
        all_scores.extend(scores)
    return np.array(all_scores)

# Set up LIME explainer
EXPLAINER = LimeTextExplainer(verbose=True)
#te = TextExplainer(n_samples=5000, random_state=42)

def predictor(texts):
    outputs = model(**tokenizer(texts, return_tensors="pt"))
    probas = F.softmax(outputs.logits).detach().numpy()
    return probas


def print_token_probs(generation_output):
    """
    Computes the transition scores of sequences given the generation scores 
    (and beam indices, if beam search was used). This is a convenient method 
    to quicky obtain the scores of the selected tokens at generation time.
    https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationMixin.compute_transition_scores
    

    Input: 
        generation_output_sequences: output.sequences from model.generate()
    
    Prints out | token | token string | logits | probability
    """
    # Detokenize
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    # Split input from response
    output_cleaned = output.split('### Response:')[1].strip()
    # Tokenize response only
    inputs = tokenizer(
        output_cleaned,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]

    # Compute transition scores
    transition_scores = model.compute_transition_scores(
        generation_output.sequences, generation_output.scores, normalize_logits=True
    )

    token_scores = []
    # Print put token, token string, logits and probability
    for tok, score in zip(input_ids[0].cpu(), transition_scores[0].cpu()):
        # | token | token string | logits | probability
        token = tokenizer.decode(tok)
        token_score = np.exp(score.numpy())
        token_scores.append([token, token_score])
        # print(f"| {tok:5d} | {token:8s} | {score.numpy():.4f} | {token_score:.2%}")
    return token_scores

def ppl(scores):
    """
    Calculate perplexity
    """
    p_W = np.prod(scores)
    # print("p_W", p_W)
    Pnorm_W = p_W ** (1/len(scores))
    # print("Pnorm_W", Pnorm_W)
    ppl = 1/Pnorm_W
    # print(ppl)
    return ppl


while True:
    instruction=input("\nEnter instruction. Press enter to exit. ")
    if len(instruction) <= 0:
        print("Breaking free!")
        break
    input_token = input("Enter optional input: ")
    output, scores, generation_output = evaluate(instruction, input_token)
    output_cleaned = output.split('### Response:')[1].strip()

    print(f"\nResponse: {output_cleaned}")
    
    print(f"\nScores:\nType{type(scores)}\nShape: {scores.shape}\nScores: {scores}")
    token_scores = print_token_probs(generation_output)

    tokens, scores = map(list, zip(*token_scores))
    perplexity = ppl(scores)
    print(f"\nPerplexity: {perplexity}")


    see_more = input("Do you want to see the whole output? y/n: ")
    if see_more == "y":
        print(f"\nWhole output string:\n{generation_output}")

    want_lime = input("Do you want LIME? y/n: ")
    if want_lime == "y":
        #num_features = input("How many features do you want in the explenation? Default is 10. ")
        # Explain predictions using LIME
        exp = EXPLAINER.explain_instance(output_cleaned, predictor, num_features=20, num_samples=2000)
        save_path = input("Where do you want to save the image? Whole path, including .html ")