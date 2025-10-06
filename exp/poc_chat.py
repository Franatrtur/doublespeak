"""
Proof of concept for bidirectional stenography.
"""


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

# --- Configuration ---
MODEL_NAME = "context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16"
TOP_P = 0.9  # Nucleus sampling probability
MAX_NEW_TOKENS = 50 # Max number of tokens to generate in this demo


class EndConfig:
    Agressive = 1
    Natural = 2
    Random = 3

ENDING_CONFIG = EndConfig.Natural


# --- 1. Load Model and Tokenizer ---
print(f"Loading model: {MODEL_NAME}...")
# Note: For a CPU-only machine, this may take a significant amount of memory.
# If you run into memory issues, consider a smaller model or a quantized version.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16, # Use float16 to reduce memory, change to float32 if issues arise
    #device_map="auto" # Automatically uses CPU if no GPU is detected
)
print("Model loaded successfully.")

# --- 2. Prepare Initial Input with Chat Template ---
# Here we define the conversation history.
# The system prompt gives the model its persona or instructions.

OPENING = "well, idk"

chat_history = [
    {"role": "system", "content": "You are Mark, my friend from college, I am Alice. We text each other in a careless way often making errors, omitting punctuation, using slang or using emojis."},
    {"role": "user", "content": "hey mark, how u doin 😅"},
    {"role": "assistant", "content": OPENING}
]



# The tokenizer will format this into the correct prompt string for the model.
# add_generation_prompt=True adds the tokens that signal the model to start generating a response.
input_ids = tokenizer.apply_chat_template(
    chat_history,
    add_generation_prompt=False,

    return_tensors="pt"
).to(model.device)

print(input_ids)
assert False

generated_ids = input_ids


hidden_message = b'Ahoj!'

message_payload = bytes([1]) + hidden_message
payload = int.from_bytes(hidden_message, byteorder='big')


# --- 3. Generation Loop with Top-P Candidates ---
print("\n--- Starting Generation ---")

# Turn off gradient calculations for faster inference
with torch.no_grad():
    for step in range(MAX_NEW_TOKENS):
        print(f"\n--- Step {step + 1} ---")

        # Get model outputs (logits)
        print(generated_ids)
        outputs = model(generated_ids)
        logits = outputs.logits

        # Get the logits for the very next token
        next_token_logits = logits[:, -1, :]

        # Convert logits to probabilities using softmax
        probabilities = F.softmax(next_token_logits, dim=-1)

        # --- Top-P (Nucleus) Filtering ---
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > TOP_P
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probabilities[:, indices_to_remove] = 0
        probabilities /= torch.sum(probabilities)

        # --- Display Viable Candidates ---
        print("Top-P Candidates (p > 0):")
        viable_tokens = []
        candidate_indices = torch.where(probabilities[0] > 0)[0]

        # CORRECTED: Store the integer ID, not the decoded string
        for token_id_tensor in candidate_indices:
            token_id_int = token_id_tensor.item()
            prob = probabilities[0, token_id_int].item()
            viable_tokens.append((token_id_int, prob)) # Store the ID itself

        # Sort by probability for selection logic
        viable_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # For display purposes, you can still decode them
        for tid, prob in viable_tokens[:10]: # Display top 10 for brevity
             print(f"Token: '{tokenizer.decode(tid)}', Probability: {prob:.4f}")

        # Calculate the number of viable tokens
        end_possible = any(tid == tokenizer.eos_token_id for tid, _ in viable_tokens)
        
        base = len(viable_tokens)
        print(f'\n{base} tokens to choose from')

        # --- CORRECTED: Selection Logic ---
        # This block now assigns an INTEGER to next_token_id in all cases
        if payload > 0 or ENDING_CONFIG == EndConfig.Random or (not end_possible and ENDING_CONFIG == EndConfig.Natural):
            index = int(payload % base) # Ensure index is an integer
            payload //= base # Use integer division
            
            # Get the integer ID from the viable_tokens list
            next_token_id = viable_tokens[index][0]

        elif ENDING_CONFIG == EndConfig.Agressive or (end_possible and ENDING_CONFIG == EndConfig.Natural):
            next_token_id = tokenizer.eos_token_id
            print('Ending!')
        else:
            raise Exception("Invalid ending config or could not determine the right ending in the steganography process.")

        # --- CORRECTED: Appending and Breaking ---
        # Now, next_token_id is guaranteed to be an integer.

        # Stop if the model generates an end-of-sequence token
        if next_token_id == tokenizer.eos_token_id: # No .item() needed
            print("\n--- End of Sequence token generated. ---")
            break

        # Convert the integer ID to a tensor just before concatenation
        next_token_tensor = torch.tensor([[next_token_id]], device=model.device)

        # Append the new token tensor to our sequence
        generated_ids = torch.cat([generated_ids, next_token_tensor], dim=-1)

        # Print the token that was chosen (decode the integer ID)
        newly_generated_text = tokenizer.decode(next_token_id)
        print(f"\n{tokenizer.decode(generated_ids[0], skip_special_tokens=False)}")


# --- 4. Print Final Result ---
final_output = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
print("\n--- Final Generated Text ---")
print(final_output)