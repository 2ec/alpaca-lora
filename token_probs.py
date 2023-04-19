import numpy as np

def print_token_probs(generation_output_sequences):
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
        print(f"| {tok:5d} | {token:8s} | {score.numpy():.4f} | {token_score:.2%}")
    return token_scores


token_scores = print_token_probs(generation_output.sequences)