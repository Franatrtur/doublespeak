"""
Proof of concept for EOS bias.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- SETUP ---
MODEL_NAME = "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16"
TOP_P = 0.95

print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded.")

eos_token_id = tokenizer.eos_token_id
print(f"EOS Token ID: {eos_token_id}, Token: '{tokenizer.decode(eos_token_id)}'")


def run_poc(eos_bias: float):
    """
    Runs the proof-of-concept generation with a specific EOS logit bias.
    """
    print("\n" + "="*80)
    print(f"  RUNNING PROOF OF CONCEPT WITH {eos_bias=}")
    print("="*80)

    prompt = "When I was little, I dreamed of being a programmer."
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    print(f"Initial Prompt: '{tokenizer.decode(input_ids[0], skip_special_tokens=True)}'")
    print("-" * 80)
    print(f"{'Step':<5} | {'Picked Token':<15} | {'In Top-P?':<12} | {'EOS Probability'}")
    print(f"{'':<5} | {'':<15} | {'(p={TOP_P})':<12} |")
    print("-" * 80)


    # Generate 30 more tokens, one by one
    for i in range(30):
        with torch.no_grad():
            outputs = model(input_ids)

        # Get logits for the very next token
        next_token_logits = outputs.logits[:, -1, :]

        # --- BIAS IS APPLIED HERE ---
        # We add the bias to the logit score for the EOS token *before* softmax.
        next_token_logits[0, eos_token_id] += eos_bias

        # Get probabilities via softmax (the bias is now "baked in")
        probabilities = F.softmax(next_token_logits, dim=-1)

        # Get the specific probability of the EOS token AFTER biasing
        eos_prob = probabilities[0, eos_token_id].item()

        # --- CHECK IF EOS IS IN THE TOP-P SET ---
        sorted_probs, sorted_indices = torch.sort(probabilities[0], descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        
        # Find the index where cumulative probability exceeds TOP_P
        nucleus_cutoff_index = torch.searchsorted(cumulative_probs, TOP_P)
        
        # Get all token IDs within that nucleus
        viable_token_ids = sorted_indices[:nucleus_cutoff_index + 1].tolist()
        
        # Check if our EOS token made the cut
        in_top_p_set = "Yes" if eos_token_id in viable_token_ids else "No"
        # --- END OF CHECK ---

        # For this demo, we still just pick the most likely token to continue
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        # Decode and print
        chosen_token_str = tokenizer.decode(next_token_id[0])
        print(f"{i+1:<5} | {chosen_token_str!r:<15} | {in_top_p_set:<12} | {eos_prob:.6f}")
        
        # Append the chosen token for the next iteration
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        if next_token_id.item() == eos_token_id:
            print("\nEOS token was generated.")
            break

    print("-" * 80)
    print(f"Final generated text:\n{tokenizer.decode(input_ids[0], skip_special_tokens=True)}")


# --- RUN THE EXPERIMENTS ---
if __name__ == "__main__":
    # First run: The original scenario with no bias
    #run_poc(eos_bias=0.0)
    
    # Second run: With a strong positive bias to make EOS a viable candidate
    run_poc(eos_bias=8.0)