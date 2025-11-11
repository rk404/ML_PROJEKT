import static_ev as ev
from data_import import load_onedrive_csv
from tokenizer import load_tiktoken

kombinacje_df = load_onedrive_csv(ev.kombinacje_url)
indeksy_df = load_onedrive_csv(ev.indeksy_url)
print("Kombinacje DataFrame:")
print(kombinacje_df.head())
print("Indeksy DataFrame:")
print(indeksy_df.head())
enc = load_tiktoken("o200K_base")  # domyślnie o200K_base, można zmienić
text =  indeksy_df.iloc[0]['indeks']  # Załóżmy, że jest kolumna 'text_column'
tokens = enc.encode(text)
print("Tekst z indeksów:", text)
print("Tokeny:", tokens)
decoded = enc.decode(tokens)