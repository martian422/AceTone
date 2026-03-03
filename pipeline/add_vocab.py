# load a pretrained tokenizer, add vocabulary and save it locally.
model_path = 'outputs/pretrained/qwen2.5-vl-3b'

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_path)

print(f'the original vocab size is {tokenizer.vocab_size}\n')

specials = {"additional_special_tokens": ["<SoT>", "<EoT>"]}
tokenizer.add_special_tokens(specials)

custom_tokens = [f"<MM{i}>" for i in range(256)]
num_added = tokenizer.add_tokens(custom_tokens)

print(f'the current vocab size is {len(tokenizer)}\n')
# check if the tokenizer can correctly operate on the custom tokens
test = tokenizer("This is a test. <SoT><MM0><MM233><EoT>")
print(test["input_ids"])
print(tokenizer.decode(test["input_ids"]))
tokenizer.save_pretrained("outputs/pretrained/custom_tokenizer")
