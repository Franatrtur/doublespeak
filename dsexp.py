
from doublespeak import DoubleSpeak

ds = DoubleSpeak(model_name="cerebras/btlm-3b-8k-base", top_p = 0.8, verbose = True, debug = True)

ds.load_model(trust_remote_code=True)

stegotext = ds.encode(b'send 0.45btc to anonymous', '''🚀 Why Gemini CLI?
🎯 Free tier: 60 requests/min and 1,000 requests/day with personal Google account''')

print(stegotext)

#print(ds.decode(stegotext, 'The following is a simple pseudocode for'))