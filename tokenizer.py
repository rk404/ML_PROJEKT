import pandas as pd
import tiktoken
data = {
    "id": [1, 2, 3],
    "tekst": [
        "To jest pierwszy przykład tekstu.",
        "Drugi tekst zawiera więcej słów niż pierwszy.",
        "Ostatni wiersz ma krótszy tekst."
    ]
}
data_set = pd.DataFrame(data)
tokenizer = tiktoken.get_encoding("o200k_base")
def tokenize_text(text):
    return tokenizer.encode(text)
data_set['tokens'] = data_set['tekst'].apply(tokenize_text)
print(data_set[['tekst', 'tokens']])
 