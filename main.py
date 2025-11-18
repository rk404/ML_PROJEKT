import sys
import io
from Funkcje.tokenizacja import tokenize_column

# Ustaw kodowanie UTF-8 dla konsoli
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import static_ev as ev
from Funkcje.data_import import load_onedrive_csv
#from tokenizer import load_tiktoken

# ramka z baza indeksow materialowych
indeksy_df = load_onedrive_csv(ev.indeksy_url)

# ramka z tłumaczeniem kombinacji materialowych
kombinacje_df = load_onedrive_csv(ev.kombinacje_url)

# usuniecie niepotrzebnych kolumn z ramki indeksy_df
indeksy_df_f = indeksy_df[['ID', 'INDEKS', 'NAZWA','KOMB_ID']]

# usuniecie niepotrzebnych kolumn z ramki kombinacje_df
kombinacje_df_f = kombinacje_df[['SEGM_ID','KOMB_ID', 'POZYCJA_SEG','SLIT_ID','SLOW_ID', 'KOD', 'OPIS', 'ZALEZNY_OD_ID','NAZWA_SLOW', 'ZALEZNY_OD_SLOW_ID']]

# usunięcie z ramki indeksy_df_f wierszy z brakującymi wartościami w kolumnie KOMB_ID
indeksy_df_f = indeksy_df_f.dropna(subset=['KOMB_ID'])


indeksy_df_f_token = tokenize_column(indeksy_df_f, column_name='NAZWA', model='gpt-3.5-turbo')

