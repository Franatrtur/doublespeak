- add params col to the table of tested models in readme
- test more base models:
 1) pick a base model from the end of readme.md
 2) test encoding decoding, using example_opening.txt all using the exp folder
 for example: `python doublespeak.py --model-name="HuggingFaceTB/SmolLM3-3B-Base" --end-bias=6 --top-p 0.75  encode -m "send 0.5btc to adress 0xe4ffd093" -o @exp/opening_code.txt
 > exp/output_smollm3b_code.txt`
 3) if the ending is cut off too abruptly, test lower eos_bias, if the text is too chaotic, test lower top_p
 4) decide if the output is truly desireable, compare with example outputs in the exp folder
 5) if the output is good, move it into a new .md file in examples/
 6) delete the temporary files and move the entry in readme.md to the tested part, include a link to the .md example file in the model name col

