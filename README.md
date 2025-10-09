# DoubleSpeak
*words in disguise* 

DoubleSpeak is an LLM steganography tool that hides secret messages within natural text.

## Getting started

```bash
git clone https://github.com/Franatrtur/doublespeak.git
# > RECOMMENDED: create a python venv <
pip install -r requirements.txt
# Ensure you have access/login to Hugging Face - if using gated models like Llama:
huggingface-cli login

python doublespeak.py --help
```

## Tested models and examples

These are all publicly available models on huggingface which can be passed as the model name into Doublespeak. See the `/examples` folder for example runs, or behind the model name. Help us test more models by [contributing](#contributing)!

| Model Name | Recommended settings | Notes |
| :--- | :--- | :--- |
| **`gpt2`** [demo](./examples/gpt2.md) | `end_bias=3`, `top_p=0.5` | Quick and small for plain text. Can get overwhelmed. |
| **`HuggingFaceTB/SmolLM3-3B-Base`** [demo](./examples/smollm3-3b.md) | Text: `end_bias=3`, `top_p=0.65` / Code: `end_bias=6`, `top_p=0.8` | Good bang for your buck. |


## Encoding and Decoding a Secret Message

You can import and use the `DoubleSpeak` class directly in your Python projects. This allows for more flexible integration than the command-line interface.
To hide a message, use the `encode` method. To retrieve it, use the `decode` method.  
Ensure you use the **exact same** initialization parameters (`model_name`, `top_p`, etc.) and the same `text_opening` for both operations. You should provide an opening to seed the context of the generated text. For better results, see our [recommendations](#recommendations).

__Successful transfer of the message is not guaranteed.__ Unfortunately. See the section on [inference determinism problems](#determinism).

### Python usage

A super fast python starter:
```python
from doublespeak import DoubleSpeak
ds = DoubleSpeak()
stegotext = ds.encode(
    hidden_message=b"This is a very secret message.",
    text_opening="The launch codes are as follows:"
)
decoded_bytes = ds.decode(
    stegotext=stegotext,
    text_opening="The launch codes are as follows:"
)
```
See the [python example](./examples/example.py) for more.

### Commandline usage

Make the script executable (only if using venv) `chmod +x doublespeak.py` or run with `python doublespeak.py`.  

The secret message is provided as bytes and is be hidden in a generated "stegotext".

Example:
```bash
#   === script ===    =========== settigns =-=========  = op =      ============== input & output ================
python doublespeak.py --model-name="gpt2" --top-p 0.45  encode -m "secret message" -o @opening_text.txt > stegotext.txt
python doublespeak.py --model-name="gpt2" --top-p 0.45  decode -s @stegotext.txt -o @opening_text.txt > secret.txt

# example using pipe:
echo "secret message" | ./doublespeak.py -o "Vacation will take place in" encode
```

### Key Configuration Options

*   `--model-name`: The Hugging Face model ID to use. (*Default:* `HuggingFaceTB/SmolLM3-3B-Base`)
*   `--top-p`: Nucleus sampling probability. Lower values increase stealth but decrease payload capacity.
*   `--end-bias`: How strongly does the model want to end the text after encoding the whole payload.

For more details run with `-h` or for example with `encode -h` for info on the encode command parameters (message, opening, ...).
There are methods `DoubleSpeak.export_settings` and `import_settings(str)` which serialize the settings of the Doublespeak object. Automatically exported in commandline usage.


#### Recommendations
 - Remember the settings (`model-name`, `opening`, `top-p`, `end-bias` and potentially `ending` and `device`) when encoding. The resulting stegotext cannot be decoded without the proper settings.
 - Speed: Any model will likely have to be downloaded the first time you use it, which can take a minute or two.
 - Use a longer opening with a good direction yet good room for creativity. Do not end the opening with a space. If the text is too chaotic, try lowering the `top-p` setting.
 - Pay attention to newlines/spaces at the end of your stegotext! They are represented as tokens too and can confuse the model or disrupt the decoding process completely. 

## How it Works

Unlike traditional text steganography that modifies existing text (e.g., changing whitespace), DoubleSpeak influences the generation process of an LLM to encode data directly into the choice of words, making the cover text statistically indistinguishable from normal model output.

Normally, an LLM generates text by calculating probabilities for the next word and randomly sampling from the best options (defined by `top_p`).

DoubleSpeak intercepts this step. Instead of rolling a die, it uses the bits of your secret message to mathematically determine which valid token to choose next. To a casual observer (or statistical analyzer not possessing the exact setup and message), the text appears to be generated randomly.

## Determinism

Generating the exact same text twice is not guaranteed. Different hardware, especially CPUs versus GPUs, will almost certainly produce different text even with identical settings. 
Different hardware (especially CPU vs. GPU) will produce different results because they handle math differently. This can cause the message to be lost.

To mitigate this, we provide the `device` setting, allowing you to specify `cpu` or `cuda`. While two different CPUs are more likely to produce the same output than a CPU and a GPU, this is still not a guarantee.

Perfect reproducibility, even on a single device, is a notoriously difficult problem, as described [this paper](https://www.rivista.ai/wp-content/uploads/2025/09/https___thinkingmachines.pdf). We plan on integrating the batch-invariant version of tensor operations into DoubleSpeak in the near future.


## Disclaimer

This tool is for educational and research purposes. Large language models were used during development.  
**Steganography hides the existence of a message, but it does not cryptographically protect its contents.** It should be treated as a concealment technique, not as a substitute for encryption.

## Contributing

We need help with development and testing, see [CONTRIBUTING.md](./CONTRIBUTING.md) for details.


