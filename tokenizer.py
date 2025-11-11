import tiktoken
tokenizer = tiktoken.get_encoding("o200k_base")
text = "To jest test tokenizera o200k_base."
tokens = tokenizer.encode(text)
print("Tokeny:", tokens)

# dtworzenie tekstu
decoded = tokenizer.decode(tokens)
print("Odtworzony tekst:", decoded)