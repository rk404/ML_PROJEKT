import static_ev as ev
from data_import import load_onedrive_csv

kombinacje_df = load_onedrive_csv(ev.kombinacje_url)
indeksy_df = load_onedrive_csv(ev.indeksy_url)
print("Kombinacje DataFrame:")
print(kombinacje_df.head())
print("Indeksy DataFrame:")
print(indeksy_df.head())
<<<<<<< HEAD

#lista kolumn w ramce danych kombinacje_df
print("Kolumny w kombinacje_df:")   
print(kombinacje_df.columns.tolist())
print("Kolumny w indeksy_df:")   
print(indeksy_df.columns.tolist())
=======
enc = load_tiktoken("o200K_base")  # domyślnie o200K_base, można zmienić
text =  indeksy_df.iloc[0]['indeks']  # Załóżmy, że jest kolumna 'text_column'
tokens = enc.encode(text)
print("Tekst z indeksów:", text)
print("Tokeny:", tokens)
decoded = enc.decode(tokens)
>>>>>>> 91388cddae77affa560c15e95e5f78255636f56f
