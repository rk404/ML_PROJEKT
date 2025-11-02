import static_ev as ev
from data_import import load_onedrive_csv

kombinacje_df = load_onedrive_csv(ev.kombinacje_url)
indeksy_df = load_onedrive_csv(ev.indeksy_url)
print("Kombinacje DataFrame:")
print(kombinacje_df.head())
print("Indeksy DataFrame:")
print(indeksy_df.head())