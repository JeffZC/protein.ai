from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

f = open("train_ds")
sequence = f.read()
tokens = tokenizer.tokenize(sequence)

tokenizer.save_pretrained("model_1")

new_tokenizer = tokenizer.train_new_from_iterator(sequence, 10)

new_tokens = new_tokenizer.tokenize(sequence)

print(new_tokens)
