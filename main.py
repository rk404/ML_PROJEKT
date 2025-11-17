import static_ev as ev
from data_import import load_onedrive_csv

kombinacje_df = load_onedrive_csv(ev.kombinacje_url)
indeksy_df = load_onedrive_csv(ev.indeksy_url)
print("Kombinacje DataFrame:")
print(kombinacje_df.head())
print("Indeksy DataFrame:")
print(indeksy_df.head())

#lista kolumn w ramce danych kombinacje_df
print("Kolumny w kombinacje_df:")   
print(kombinacje_df.columns.tolist())
print("Kolumny w indeksy_df:")   
print(indeksy_df.columns.tolist())
