#!venv/bin/python

from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import argparse
import sys
import time

VERSION = '0.1'

Token = Any

def require_loaded(method):
    """Decorator to ensure the model is loaded before a method is called."""
    def wrapper(self, *args, **kwargs):
        if not self.loaded:
            self.load_model()
        return method(self, *args, **kwargs)
    return wrapper

def progress_bar(name: str, iteration: int, total: int, length=40, refuse_finish: bool = False):
    
    percent = 100 * (iteration / float(total))
    
    filled_length = int(length * iteration // total)
    bar = '■' * filled_length + '-' * (length - filled_length)
    
    extended_name = name + ": " + ' ' * (12 - len(name))
    
    print(f'\r{extended_name}[{bar}] {percent:.0f}% Complete', end='\r', file=sys.stderr)
    
    if iteration == total and not refuse_finish:
        print(file=sys.stderr)     # flush


class DoubleSpeak:

    # --- Configuration Enums ---
    AgressiveEnd = 1
    NaturalEnd = 2
    RandomEnd = 3

    # --- Internal Return Codes ---
    EndIndex = -1
    InvalidIndex = -2

    def __init__(self,
            model_name="context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16",
            top_p=0.9,
            ending=0,
            end_bias=7.0,
            verbose=False,
            debug=False
        ):
        self.model_name = model_name
        self.top_p = top_p
        self.ending = ending if ending != 0 else DoubleSpeak.NaturalEnd
        self.end_bias = end_bias
        self.verbose = verbose
        self.debug = debug
        self.loaded = False

    def load_model(self, **kwargs):
        """Loads the model and tokenizer from Hugging Face."""
        if self.verbose or self.debug:
            print(f"Loading doublespeak model: {self.model_name}", file=sys.stderr)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=device,
            **kwargs
        )
        self.device = self.model.device
        
        if self.verbose or self.debug:
            print(f"Model loaded successfully on {self.device}.", file=sys.stderr)

        self.loaded = True

    def chat_respond(self, chat_history: list[dict[str, str]], hidden_message: bytes, text_opening: str = ''):
        """
        (Not Implemented) A future method to embed a hidden message within a natural
        chat response based on conversation history.
        """
        pass

    @require_loaded
    def encode(self, hidden_message: bytes, text_opening: str = '') -> str:
        """
        Encodes a byte string into a natural-looking text (stegotext).

        Args:
            hidden_message: The bytes to hide.
            text_opening: An optional string to start the generation process.

        Returns:
            The generated text containing the hidden message.
        """
        starting_tokens_ids = self.tokenizer(
            text_opening,
            return_tensors="pt"
        ).input_ids.to(self.device)

        complete_sequence_ids = self._encode_bytes(starting_tokens_ids, hidden_message)

        return self.tokenizer.decode(complete_sequence_ids[0], skip_special_tokens=True)

    @require_loaded
    def decode(self, stegotext: str, text_opening: str = '') -> bytes:
        """
        Decodes a hidden byte string from a stegotext.

        Args:
            stegotext: The text containing the hidden message.
            text_opening: The same starting string used for encoding.

        Returns:
            The decoded byte string.
        """
        starting_tokens_ids = self.tokenizer(
            text_opening,
            return_tensors="pt"
        ).input_ids.to(self.device)

        token_ids_sequence = self.tokenizer(
            stegotext,
            return_tensors="pt"
        ).input_ids.to(self.device)

        hidden_message = self._decode_bytes(starting_tokens_ids, token_ids_sequence)

        return hidden_message

    def _encode_bytes(self, starting_tokens_ids, hidden_message: bytes) -> torch.Tensor:
        """The core encoding loop, now with KV Caching for performance."""
        if self.debug:
            print(f'DoubleSpeak encoding message {hidden_message} with {self=}', file=sys.stderr)

        # --- State Management for KV Caching ---
        past_key_values = None
        # This holds the complete sequence for the final output
        token_ids_sequence = starting_tokens_ids
        # This will hold the input for the *next* model call.
        # It starts as the full opening text, but becomes a single token in the loop.
        current_input_ids = starting_tokens_ids
        
        hidden_payload = int.from_bytes(bytes([1]) + hidden_message, byteorder='big')
        total_payload_bitlength = hidden_payload.bit_length()

        while True:
            extending = hidden_payload == 0

            # --- 1. Get next token predictions and updated cache ---
            # On the first pass, current_input_ids is the full prompt.
            # On subsequent passes, it's only the most recently generated token.
            viable_tokens, end_possible, past_key_values = self._next_tokens(current_input_ids, past_key_values)

            # --- 2. Decision Logic: To End or To Continue? ---
            should_end = self.ending == self.AgressiveEnd or (end_possible and self.ending == self.NaturalEnd)
            must_continue = hidden_payload > 0 or self.ending == self.RandomEnd
            
            if not must_continue and should_end:
                next_token_id = self.tokenizer.eos_token_id
                if self.debug:
                    print(f"_encode_bytes: Payload is zero and EOS is a natural choice. Ending generation.", file=sys.stderr)
                # Append the final token and break the loop
                next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
                token_ids_sequence = torch.cat([token_ids_sequence, next_token_tensor], dim=-1)
                break
            
            # --- 3. Continue Encoding ---
            base = len(viable_tokens)
            if base == 0:
                # This should now be extremely rare, only occurring if the model's entire output
                # distribution is zero, which would indicate a deeper issue.
                raise RuntimeError('_encode_bytes: Must continue encoding, but no viable tokens were found at all!')
            
            index = hidden_payload % base
            hidden_payload //= base
            next_token_id = viable_tokens[index][0]

            if self.debug:
                print(f'_encode_bytes: Chose token {repr(self.tokenizer.decode(next_token_id))} at index {index} from {base} options. Remaining {hidden_payload.bit_length()=}, {end_possible=}', file=sys.stderr)

            # --- 4. Update State for the Next Loop Iteration ---
            next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
            # Append the chosen token to our complete history for the final output
            token_ids_sequence = torch.cat([token_ids_sequence, next_token_tensor], dim=-1)
            # The input for the *next* iteration is ONLY the token we just generated.
            current_input_ids = next_token_tensor
            
            if self.debug:
                print(f"Current sequence: ...{self.tokenizer.decode(token_ids_sequence[0][-20:])}", file=sys.stderr)

            if not extending and self.verbose and not self.debug:
                progress_bar('Encoding message', total_payload_bitlength - hidden_payload.bit_length(), total_payload_bitlength)

        assert hidden_payload == 0, '_encode_bytes: Not all of the payload was encoded.'
        return token_ids_sequence

    def _decode_bytes(self, starting_tokens_ids, token_ids_sequence) -> bytes:
        """The core decoding loop, now with KV Caching for performance."""
        decoded_digits = []
        
        # --- State Management for KV Caching ---
        past_key_values = None
        current_input_ids = starting_tokens_ids
        
        start_index = len(starting_tokens_ids[0])
        total_tokens = len(token_ids_sequence[0])

        for i in range(start_index, total_tokens):
            if self.debug:
                print(f'_decode_bytes: Decoding token {i-start_index+1}/{total_tokens-start_index}', file=sys.stderr)
            elif self.verbose:
                progress_bar('Decoding tokens', i-start_index+1, total_tokens-start_index)
            
            # Get the viable tokens based on all prior context
            viable_tokens, _, past_key_values = self._next_tokens(current_input_ids, past_key_values)
            base = len(viable_tokens)
            
            next_token_id = token_ids_sequence[0, i].item()

            if next_token_id == self.tokenizer.eos_token_id:
                break # Valid end, stop decoding

            if base == 0:
                 raise ValueError(f'Invalid stegotext: No viable tokens could have followed the sequence at index {i}.')

            # Find the index (digit) of the actual next token within the viable choices
            found_index = -1
            for index, (tid, _) in enumerate(viable_tokens):
                if tid == next_token_id:
                    found_index = index
                    break
            
            if found_index == self.InvalidIndex:
                raise ValueError(f'Invalid stegotext: Could not decode token ID {next_token_id} at index {i}.')
            
            digit = found_index
            decoded_digits.append((digit, base))
            
            # The input for the next iteration is the token we just decoded
            current_input_ids = token_ids_sequence[:, i:i+1]

        # Reconstruct the original integer from the mixed-radix representation
        decoded_payload = 0
        for digit, base in reversed(decoded_digits):
            decoded_payload *= base
            decoded_payload += digit
        
        if decoded_payload == 0:
            return b''
            
        byte_length = (decoded_payload.bit_length() + 7) // 8
        return decoded_payload.to_bytes(byte_length, byteorder='big')[1:]

    @require_loaded
    def _next_tokens(self, input_ids, past_key_values=None) -> tuple[list[tuple[int, float]], bool, tuple]:
        """
        Calculates the next set of viable tokens using KV caching.
        
        Returns:
            A tuple containing:
            - A list of (token_id, probability) for viable non-EOS tokens.
            - A boolean indicating if the EOS token was a viable choice.
            - The updated past_key_values (KV cache) for the next iteration.
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        logits = outputs.logits[:, -1, :]

        # --- Step 1: Get the NATURAL probabilities first for sorting later ---
        natural_probabilities = F.softmax(logits, dim=-1)

        # --- Step 2: Create a biased version of the logits for the EOS check ---
        biased_logits = logits.clone()
        if self.end_bias != 0.0:
            biased_logits[0, self.tokenizer.eos_token_id] += self.end_bias

        # --- Step 3: Use the BIASED probabilities for nucleus sampling ---
        biased_probabilities = F.softmax(biased_logits, dim=-1)
        
        sorted_probs, sorted_indices = torch.sort(biased_probabilities, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        
        viable_mask = torch.ones_like(biased_probabilities[0], dtype=torch.bool)
        viable_mask[indices_to_remove] = False
        
        # --- Step 4: Perform the checks and build the final list ---
        candidate_indices = torch.where(viable_mask)[0]
        
        eos_was_viable = self.tokenizer.eos_token_id in candidate_indices

        viable_tokens = []
        for token_id_tensor in candidate_indices:
            token_id_int = token_id_tensor.item()
            if token_id_int == self.tokenizer.eos_token_id:
                continue
            
            natural_prob = natural_probabilities[0, token_id_int].item()
            viable_tokens.append((token_id_int, natural_prob))

        # If the only viable token after top_p sampling was the EOS token,
        # but we must continue encoding, we need an alternative.
        # This block finds the single most probable non-EOS token from the entire
        # distribution and uses it as the only option.
        if not viable_tokens and eos_was_viable:
            if self.debug:
                print("_next_tokens: Only EOS was viable, finding next best token to continue encoding.", file=sys.stderr)
            
            # We will use the already sorted indices from the biased probabilities.
            # The relative order of non-EOS tokens is the same as in the natural distribution.
            for token_id_tensor in sorted_indices[0]:
                token_id_int = token_id_tensor.item()
                if token_id_int != self.tokenizer.eos_token_id:
                    # This is the most probable token that is not EOS.
                    natural_prob = natural_probabilities[0, token_id_int].item()
                    viable_tokens.append((token_id_int, natural_prob))
                    break # We only need the single best one.
        # --- END OF FIX ---
        
        viable_tokens.sort(key=lambda x: x[1], reverse=True)

        if len(viable_tokens) > 0:
            eos_token_rank = (sorted_indices[0] == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0].item()
            eos_ratio = eos_token_rank / len(viable_tokens) if len(viable_tokens) > 0 else float('inf')
            eos_probability = max(100 * (0.5 / (eos_ratio - 0.5)) if eos_ratio > 0.5 else 100, 100)

            if self.verbose and not self.debug:
                print(f'\rEnding availability: {(eos_probability):.0f}% ===', end='\r', file=sys.stderr)

        return viable_tokens, eos_was_viable, outputs.past_key_values


    def __str__(self):
        return f'DoubleSpeak({self.model_name=},{self.top_p=},{self.ending=},{self.end_bias=},{self.verbose=},{self.debug=}) <{self.loaded=}>'

    def __repr__(self):
        return str(self)


def main():
    """
    Command-line interface for the DoubleSpeak steganography tool.

    This function parses command-line arguments to control the encoding and
    decoding processes, instantiates the DoubleSpeak class, and prints the
    result to standard output or a specified file.
    """
    # Define ANSI color codes for the logo
    ORANGE = '\033[38;2;255;165;0m'
    RESET = '\033[0m'

    parser = argparse.ArgumentParser(
        description="""
        DoubleSpeak: A steganography tool using large language models.

        This tool can either 'encode' a hidden message into a carrier text,
        or 'decode' a hidden message from such a text.

        Inputs can be provided as a direct string, from a file by prefixing
        the path with '@' (e.g., '@message.txt'), or from stdin if the
        primary input argument is omitted.
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- Global arguments for the DoubleSpeak class configuration ---
    parser.add_argument(
        '--model-name',
        type=str,
        default="context-labs/meta-llama-Llama-3.2-3B-Instruct-FP16",
        help="The Hugging Face model identifier to use.\n(default: %(default)s)"
    )
    # ... (all other global arguments like --top-p, --ending, --verbose, etc. remain the same)
    parser.add_argument(
        '--top-p', type=float, default=0.9,
        help="Nucleus sampling probability. (default: %(default)s)"
    )
    parser.add_argument(
        '--ending', choices=['aggressive', 'natural', 'random'], default='natural',
        help="Strategy for ending the generated text.\n(default: %(default)s)"
    )
    parser.add_argument(
        '--end-bias', type=float, default=7.0,
        help="A bias added to the logit of the EOS token.\n(default: %(default)s)"
    )
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help="Disable verbose logging (e.g., progress bars) to stderr."
    )
    parser.add_argument(
        '--debug', action='store_true',
        help="Enable detailed debug logging to stderr."
    )


    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    # --- 'encode' command ---
    parser_encode = subparsers.add_parser(
        'encode', help="Encode a hidden message into generated text."
    )
    parser_encode.add_argument(
        '-m', '--hidden-message',
        dest='source',
        type=str,
        help="The secret message to encode. Prefix with '@' to read from a file.\nIf omitted, reads from stdin."
    )
    parser_encode.add_argument(
        '-o', '--opening',
        dest='text_opening',
        type=str,
        default='',
        help="Optional text to seed the generation. Prefix with '@' to read from a file."
    )
    parser_encode.add_argument(
        '-O', '--output',
        type=argparse.FileType('w', encoding='utf-8'),
        default=sys.stdout,
        help="Output file for the generated stegotext. Defaults to stdout."
    )

    # --- 'decode' command ---
    parser_decode = subparsers.add_parser(
        'decode', help="Decode a hidden message from stegotext."
    )
    parser_decode.add_argument(
        '-s', '--stegotext',
        dest='source',
        type=str,
        help="The stegotext to decode. Prefix with '@' to read from a file.\nIf omitted, reads from stdin."
    )
    parser_decode.add_argument(
        '-o', '--opening',
        dest='text_opening',
        type=str,
        default='',
        help="The same optional seeding text used during encoding. Prefix with '@' to read."
    )
    parser_decode.add_argument(
        '-O', '--output',
        type=argparse.FileType('w', encoding='utf-8'),
        default=sys.stdout,
        help="Output file for the decoded message. Defaults to stdout."
    )

    args = parser.parse_args()
    
    # --- Helper function to resolve input source ---
    def get_content_from_source(source_str: str) -> str:
        """Reads content from a string, a file path prefixed with '@', or stdin."""
        if source_str is None:
             # Check if we're trying to read from a non-blocking stdin
            if sys.stdin.isatty():
                print(f"Enter content and press Ctrl+D to finish:", file=sys.stderr)
            return sys.stdin.read()
        if source_str.startswith('@'):
            filepath = source_str[1:]
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return f.read()
            except FileNotFoundError:
                print(f"Error: Input file not found at '{filepath}'", file=sys.stderr)
                sys.exit(1)
        return source_str


    if not args.quiet or args.debug:
        try:
            from pathlib import Path
            logo_path = Path(__file__).parent / 'logo.txt'
            with open(logo_path, 'r') as f:
                logo = f.read()
            print(f"{ORANGE}{logo}{RESET}", file=sys.stderr)
        except (FileNotFoundError, NameError):
            print('DS', file=sys.stderr)
        print(f"{ORANGE}--- DoubleSpeak v{VERSION} ---{RESET}", file=sys.stderr)

    # --- Get content for all inputs ---
    main_content = get_content_from_source(args.source)
    opening_content = get_content_from_source(args.text_opening) if args.text_opening else ""

    # Map the string choice for 'ending' to the class integer constants
    ending_map = {
        'aggressive': DoubleSpeak.AgressiveEnd,
        'natural': DoubleSpeak.NaturalEnd,
        'random': DoubleSpeak.RandomEnd
    }

    try:
        ds = DoubleSpeak(
            model_name = args.model_name,
            top_p = args.top_p,
            ending = ending_map[args.ending],
            end_bias = args.end_bias,
            verbose = not args.quiet,
            debug = args.debug
        )

        with args.output as output_file:
            if args.command == 'encode':
                hidden_message_bytes = main_content.encode('utf-8')
                stegotext = ds.encode(
                    hidden_message=hidden_message_bytes,
                    text_opening=opening_content
                )
                output_file.write(stegotext)
                print("", file=sys.stderr)

            elif args.command == 'decode':
                decoded_bytes = ds.decode(
                    stegotext=main_content,
                    text_opening=opening_content
                )
                decoded_message = decoded_bytes.decode('utf-8', errors='replace')
                output_file.write(decoded_message)
                print("", file=sys.stderr)

    except (ValueError, RuntimeError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    except torch.cuda.OutOfMemoryError:
        print("Error: CUDA out of memory. Try a smaller model.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()