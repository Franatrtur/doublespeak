from .doublespeak import EndConfig

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16")
messages = [
    {"role": "user", "content": "Who are you?"},
]
print(pipe(messages))

# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16")
# model = AutoModelForCausalLM.from_pretrained("context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16")