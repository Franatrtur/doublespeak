from doublespeak import DoubleSpeak

# Initialize the class with your desired settings
ds = DoubleSpeak(
    model_name="gpt2",
    top_p=0.5,
    end_bias=3.0,
    verbose=True # Shows progress bars
)

# The model isn't loaded until the first encode/decode call.
# You can also load it manually if needed:
# ds.load_model()

# The message to hide (must be bytes)
hidden_message_bytes = b"This is a very secret message."

# The opening text that guides the generation
opening_text = "The launch codes are as follows:"

# --- ENCODING ---
print("Encoding message...")
stegotext = ds.encode(
    hidden_message=hidden_message_bytes,
    text_opening=opening_text
)

print("\n--- Generated Stegotext ---")
print(stegotext)
print("---------------------------\n")


# --- DECODING ---
print("Decoding message from stegotext...")
decoded_bytes = ds.decode(
    stegotext=stegotext,
    text_opening=opening_text
)

decoded_message = decoded_bytes.decode('utf-8')

print("\n--- Decoded Message ---")
print(decoded_message)
print("-----------------------\n")

# Verify the result
assert hidden_message_bytes == decoded_bytes
print("Success! The decoded message matches the original.")