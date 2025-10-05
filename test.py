
import unittest
from doublespeak import DoubleSpeak
import os

class TestSmollLM3B(unittest.TestCase):

    def test_encode_decode_article(self):
        # Parameters from examples/smollm3-3b.md
        model_name = "HuggingFaceTB/SmolLM3-3B-Base"
        end_bias = 3
        top_p = 0.65
        ending = "natural"
        secret_message_str = "send 0.5btc to adress 0xe4ffd093"
        secret_message_bytes = secret_message_str.encode('utf-8')

        # Map string ending to enum
        ending_map = {
            'natural': DoubleSpeak.NaturalEnd,
        }

        ds = DoubleSpeak(
            model_name=model_name,
            top_p=top_p,
            ending=ending_map[ending],
            end_bias=end_bias,
            verbose=False
        )

        with open("./test/opening_article.txt", "r") as f:
            opening_text = f.read()

        with open("./test/output_smollm3b_article.txt", "r") as f:
            expected_stegotext = f.read()
        
        # 1. Test encoding
        stegotext = ds.encode(hidden_message=secret_message_bytes, text_opening=opening_text)
        self.assertEqual(stegotext.strip(), expected_stegotext.strip())

        # 2. Test decoding
        decoded_message = ds.decode(stegotext=stegotext, text_opening=opening_text)
        self.assertEqual(decoded_message, secret_message_bytes)

    def test_encode_decode_code(self):
        # Parameters from examples/smollm3-3b.md
        model_name = "HuggingFaceTB/SmolLM3-3B-Base"
        end_bias = 6
        top_p = 0.75
        ending = "natural"
        secret_message_str = "send 0.5btc to adress 0xe4ffd093"
        secret_message_bytes = secret_message_str.encode('utf-8')
        self.maxDiff = None

        # Map string ending to enum
        ending_map = {
            'natural': DoubleSpeak.NaturalEnd,
        }

        ds = DoubleSpeak(
            model_name=model_name,
            top_p=top_p,
            ending=ending_map[ending],
            end_bias=end_bias,
            verbose=False
        )

        with open("./test/opening_code.txt", "r") as f:
            opening_text = f.read()

        with open("./test/output_smollm3b_code.txt", "r") as f:
            expected_stegotext = f.read()

        # 1. Test encoding
        stegotext = ds.encode(hidden_message=secret_message_bytes, text_opening=opening_text)
        self.assertEqual(stegotext.strip(), expected_stegotext.strip())

        # 2. Test decoding
        decoded_message = ds.decode(stegotext=stegotext, text_opening=opening_text)
        self.assertEqual(decoded_message, secret_message_bytes)


if __name__ == '__main__':
    unittest.main()
