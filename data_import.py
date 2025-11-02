import requests
import pandas as pd
from io import StringIO

def load_onedrive_csv(share_url):
    """
    Wczytuje CSV z publicznego linku OneDrive bezpośrednio do DataFrame
    """
    # Dodaj parametr download do URL
    if '?' in share_url:
        download_url = share_url + '&download=1'
    else:
        download_url = share_url + '?download=1'
    
    print(f"Pobieranie danych z OneDrive...")
    response = requests.get(download_url)
    
    if response.status_code == 200:
        # Dekoduj content i wczytaj do pandas
        csv_content = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(csv_content), sep=';')
        print(f"Wczytano {len(df)} wierszy i {len(df.columns)} kolumn")
        return df
    else:
        print(f"Błąd pobierania: {response.status_code}")
        return None

# Użycie:
url = "https://1drv.ms/x/c/8E1FC8AC1722703E/EWTt4HgC3xNItsgAxOL2w3kBqsaS3duIJzIleeU279QTKA?e=573HeJ"
df = load_onedrive_csv(url)

if df is not None:
    print("\nPierwsze 5 wierszy:")
    print(df.head())
    print("\nInformacje o DataFrame:")
    print(df.info())