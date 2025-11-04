import requests
import pandas as pd
from io import StringIO
import urllib3

# Wyłącz ostrzeżenia o niezweryfikowanym SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_onedrive_csv(share_url):
    '''
    Wczytuje CSV z publicznego linku OneDrive bezpośrednio do DataFrame
    '''
    # Dodaj parametr download do URL
    if '?' in share_url:
        download_url = share_url + '&download=1'
    else:
        download_url = share_url + '?download=1'

    print('Pobieranie danych z OneDrive...')
    # Wyłącz weryfikację SSL (verify=False)
    response = requests.get(download_url, verify=False)

    if response.status_code == 200:
        # Dekoduj content i wczytaj do pandas
        csv_content = response.content.decode('utf-8')
        df = pd.read_csv(StringIO(csv_content), sep=';')
        print(f'Wczytano {len(df)} wierszy i {len(df.columns)} kolumn')
        return df
    else:
        print(f'Błąd pobierania: {response.status_code}')
        return None
