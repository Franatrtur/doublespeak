# DoubleSpeak
*words in disguise* 

DoubleSpeak is an LLm steganography tool that hides secret messages within natural text.

## Getting started

```bash
git clone https://github.com/Franatrtur/doublespeak.git
# > RECOMMENDED: create a python venv <
pip install requirements.txt
# Ensure you have access/login to Hugging Face - if using gated models like Llama:
huggingface-cli login

python doublespeak.py --help
```

#### Requirements:

*   Python 3.8+
*   PyTorch (with CUDA recommended for speed)
*   Transformers library from huggingface

## Tested models

These are all publicly available models on huggingface which can be passed as the model name into Doublespeak. See the `/examples` folder for example runs, or behind the model name. Help us test more models by [contributing](#contributing)!

| Model Name | Recommended settings | Notes |
| :--- | :--- | :--- |
| **`gpt2`** [demo](./examples/gpt2.md) | `end_bias=3`, `top_p=0.5` | Quick and small for plain text. Can get overwhelmed. |
| **`HuggingFaceTB/SmolLM3-3B-Base`** [demo](./examples/smollm3-3b.md) | Text: `end_bias=3`, `top_p=0.65` / Code: `end_bias=6`, `top_p=0.8` | Good bang for your buck. |


## Usage

Make the script executable (only if using venv) `chmod +x doublespeak.py` or run with `python doublespeak.py`.

### Encoding a Message

To hide a message, use the `encode` command. You should provide an `--opening` to seed the context of the generated text.

#### Recommendations
 - Remember the settings (`model-name`, `opening`, `top-p`, `end-bias` and potentially `ending`) when encoding. The resulting stegotext cannot be decoded without the proper settings.
 - Speed: Any model will likely have to be downloaded the first time you use it, which can take a minute or two.
 - Use a longer opening with a good direction yet good room for creativity. Do not end the opening with a space. If the text is too chaotic, try lowering the `top-p` setting.
 - Pay attention to newlines/spaces at the end of your stegotext! They are represented as tokens too and can confuse the model or disrupt the decoding process completely. 

**Example:** Hide "Meet me at dawn" starting with the text "Regarding the plans for tomorrow," and save to `stegotext.txt`.

```bash
./doublespeak.py encode -m "Meet me at dawn" -o "Regarding the plans for tomorrow," > stegotext.txt
```

Using pipes and stdin:
```bash
echo "Super secret data" | ./doublespeak.py encode > cover.txt
```

### Decoding a Message

To recover the message, use the `decode` command. **Crucial:** You must use the exact same settings: `--model-name`, `--top-p`, and `--opening` text that were used during encoding.

**Example:** Decode the file created above.

```bash
./doublespeak.py decode \
  -s @stegotext.txt \
  -o "Regarding the plans for tomorrow,"
# Output: Meet me at dawn
```

## Key Configuration Options

*   `--model-name`: The Hugging Face model ID to use. (*Default:* `HuggingFaceTB/SmolLM3-3B-Base`)
*   `--top-p`: Nucleus sampling probability. Lower values increase stealth but decrease payload capacity.
*   `--ending`: Strategy for finishing the text (`natural`, `aggressive`, `random`).

For more details run with `-h` or for example with `encode -h` for info on the encode command parameters (message, opening, ...).

## How it Works


Unlike traditional text steganography that modifies existing text (e.g., changing whitespace), DoubleSpeak influences the generation process of an LLM to encode data directly into the choice of words, making the cover text statistically indistinguishable from normal model output.

Normally, an LLM generates text by calculating probabilities for the next word and randomly sampling from the best options (defined by `top_p`).

DoubleSpeak intercepts this step. Instead of rolling a die, it uses the bits of your secret message to mathematically determine which valid token to choose next. To a casual observer (or statistical analyzer not possessing the exact setup and message), the text appears to be generated randomly.

## Disclaimer

This tool is for educational and research purposes. Large language models were used during development. The security of the hidden message depends on the secrecy of the parameters used (model, opening text, top_p settings).

## Contributing

We need help with development and testing, see [CONTRIBUTING.md](./CONTRIBUTING.md) for details.