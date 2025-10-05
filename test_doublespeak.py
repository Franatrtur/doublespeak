
import unittest
from doublespeak import DoubleSpeak

class TestDoubleSpeak(unittest.TestCase):

    def test_encode_decode_simple(self):
        ds = DoubleSpeak(model_name="gpt2", verbose=False)
        original_message = b"test message"
        stegotext = ds.encode(hidden_message=original_message)
        decoded_message = ds.decode(stegotext)
        self.assertEqual(original_message, decoded_message)

if __name__ == '__main__':
    unittest.main()
