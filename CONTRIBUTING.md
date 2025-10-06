# Contributing to this project

## 1) Code
Make a fork and submit a pull request

## 2) Other notes and questions
Open a github issue.

## 3) Testing new models

Here we present some models to test in the future:

### Open-Source Base LLMs (<6B Parameters)

| # | Model name (short) | Hugging Face ID (exact) | Params (approx) | License / notes |
|---:|---|---|---:|---|
| 1 | **OpenLLaMA 3B (v2)** | `openlm-research/open_llama_3b_v2` | ~3B | Open reproduction of LLaMA weights (base pretrain). Good general-purpose small model. |
| 2 | **RedPajama INCITE — Base 3B** | `togethercomputer/RedPajama-INCITE-Base-3B-v1` | ~3B | Apache-2.0 / community reproduction targeting LLM research. Strong baseline for small LLMs. |
| 3 | **Mistral 3B (edge / small)** | `qualcomm/Mistral-3B` | ~3B | Mistral-family small model; efficient and competitive for its size. |
| 4 | **Google — Gemma 2 (2B)** | `google/gemma-2b` | ~2B | Pretrained Gemma family weights; modern training recipes from Google (base weights available). |
| 5 | **OPT 2.7B (Meta / Facebook)** | `facebook/opt-2.7b` | ~2.7B | Classic open OPT family (base pretraining). Apache-like / intended for research. Solid baseline. |
| 6 | **Cerebras-GPT 2.7B** | `cerebras/Cerebras-GPT-2.7B` | ~2.7B | Cerebras release of GPT-style models (various sizes) with HF weights for research. |
| 7 | **Pythia 2.8B (EleutherAI)** | `EleutherAI/pythia-2.8b` | ~2.8B | Designed for research/interpretability (Pythia suite). Many community resources and checkpoints. |
| 8 | **BLOOM 1.1B** | `bigscience/bloom-1b1` | ~1.1B | BLOOM family (multilingual). Good if you want a lighter, multilingual base model. |
| 9 | **Salesforce CodeGen 1–2B (code-focused base)** | `salesforce/codegen-2-1B` | ~1B–2B | Strong base code-generation models if your focus is code. HF pages available for each size. |
| 10 | **Gemma / other 1–3B variants (summary row)** | e.g. `google/gemma-2b`, `mistralai/Voxtral-Mini-3B-2507` | 1–3B | Many modern families publish smaller base variants (Gemma, Mistral, Voxtral/Mistral forks). |

---

### Notes

- Only **base/pretrained** models (no instruct/chat fine-tunes) are included.  
- Parameter counts are approximate — check each Hugging Face card for details.  
- Most are **Apache-2.0** or similar research-friendly licenses.  
- For local inference, any of these can be quantized (GGUF, GPTQ, etc.).  

And more from gemini suggestions:

Here is a comprehensive list of other compact Large Language Models (LLMs) that could be used as alternatives to your default model. The tables below present a selection of base and instruct models, ranging from approximately 1 billion to 5 billion parameters.

### Base Models

Base models are generally suitable for the primary task of steganography where the goal is to embed data within a generated text without necessarily following instructions.

| Model Name | Parameters | Description |
| :--- | :--- | :--- |
| **`meta-llama/Llama-3.2-3B`** | 3B | A multilingual model from Meta, it has strong performance in various language tasks and could provide a solid foundation for text generation. |
| **`Qwen/Qwen2.5-3B`** | 3B | Part of the Qwen2 series by Alibaba Cloud, this model boasts extended context capabilities and supports a massive 128k token input. |
| **`microsoft/phi-3.5-mini-3.8b`** | 3.8B | A "tiny but mighty" model from Microsoft, it is optimized for reasoning and code generation and has shown strong performance compared to models twice its size. |
| **`google/gemma-3-4b`** | 4B | Developed by Google DeepMind, this is a lightweight and powerful multilingual model. |
| **`HuggingFaceTB/SmolLMv3-3B`** | 3B | An open-source model from Hugging Face designed to be a strong reasoner at a small scale, with open training details and checkpoints. |
| **`cerebras/Cerebras-GPT-2.7B`** | 2.7B | Designed for efficiency, this model follows Chinchilla scaling laws, making it compute-efficient for its size. |
| **`stabilityai/stablelm-3b-4e1t`** | 3B | A 3 billion parameter model from Stability AI, trained on a diverse dataset. |
| **`EleutherAI/gpt-neo-1.3B`** | 1.3B | A family of GPT-like models developed by EleutherAI, trained on the Pile dataset. |

### Instruct Models

Instruct models are fine-tuned to follow specific instructions, which could be beneficial for a future bidirectional steganography implementation where two parties might need to issue commands to the model.

| Model Name | Parameters | Description |
| :--- | :--- | :--- |
| **`google/gemma-3-4b-it`** | 4B | An instruction-tuned, multimodal version of the Gemma 3 4b model, capable of handling both text and image inputs to generate high-quality text outputs. |
| **`Qwen/Qwen2.5-3B-Instruct`** | 3B | An instruction-tuned version of the Qwen2.5-3B model, designed for stronger performance in non-thinking mode and better alignment with user preferences. |
| **`HuggingFaceTB/SmolLM-1.7B-Instruct`** | 1.7B | The instruct version of the SmolLM-1.7B model, fine-tuned on publicly available datasets to better follow instructions and stay on topic. |