import tiktoken
import pandas as pd
import os
import ssl
import certifi
import numpy as np

# Wyłącz weryfikację SSL dla tiktoken
os.environ['TIKTOKEN_CACHE_DIR'] = os.path.join(os.path.expanduser('~'), '.tiktoken_cache')
os.environ['SSL_CERT_FILE'] = certifi.where()

# Alternatywnie - wyłącz całkowicie weryfikację SSL
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

class SSLContextAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        context = create_urllib3_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        kwargs['ssl_context'] = context
        return super().init_poolmanager(*args, **kwargs)

# Monkey patch dla tiktoken
original_get = requests.get
def patched_get(url, **kwargs):
    if 'openaipublic.blob.core.windows.net' in url:
        kwargs['verify'] = False
    return original_get(url, **kwargs)

requests.get = patched_get

def tokenize_column(df, column_name='NAZWA', model='gpt-3.5-turbo'):
    """
    Tokenizuje tekst w wybranej kolumnie DataFrame używając tiktoken
    
    Args:
        df: DataFrame z danymi
        column_name: nazwa kolumny do tokenizacji (domyślnie 'NAZWA')
        model: model GPT do wyboru encodera (domyślnie 'gpt-3.5-turbo')
    
    Returns:
        DataFrame z dodatkową kolumną zawierającą tokeny i liczbę tokenów
    """
    # Załaduj odpowiedni encoder dla modelu
    encoding = tiktoken.encoding_for_model(model)
    
    # Utwórz kopię DataFrame
    df_copy = df.copy()
    
    # Funkcja tokenizująca pojedynczy tekst
    def tokenize_text(text):
        if pd.isna(text):
            return []
        return encoding.encode(str(text))
    
    # Tokenizuj kolumnę
    df_copy[f'{column_name}_TOKENS'] = df_copy[column_name].apply(tokenize_text)
    
    # Dodaj kolumnę z liczbą tokenów
    df_copy[f'{column_name}_TOKEN_COUNT'] = df_copy[f'{column_name}_TOKENS'].apply(len)
    
    print(f'Tokenizacja zakończona. Średnia liczba tokenów: {df_copy[f"{column_name}_TOKEN_COUNT"].mean():.2f}')
    print(f'Maksymalna liczba tokenów: {df_copy[f"{column_name}_TOKEN_COUNT"].max()}')
    print(f'Minimalna liczba tokenów: {df_copy[f"{column_name}_TOKEN_COUNT"].min()}')
    
    return df_copy


def decode_tokens(tokens, model='gpt-3.5-turbo'):
    """
    Dekoduje tokeny z powrotem na tekst
    
    Args:
        tokens: lista tokenów
        model: model GPT do wyboru encodera
    
    Returns:
        zdekodowany tekst
    """
    encoding = tiktoken.encoding_for_model(model)
    return encoding.decode(tokens)


def get_token_statistics(df, column_name='NAZWA'):
    """
    Zwraca statystyki tokenizacji dla kolumny
    
    Args:
        df: DataFrame z tokenizowanymi danymi
        column_name: nazwa bazowej kolumny
    
    Returns:
        słownik ze statystykami
    """
    token_count_col = f'{column_name}_token_count'
    
    if token_count_col not in df.columns:
        print(f'Kolumna {token_count_col} nie istnieje. Najpierw wykonaj tokenizację.')
        return None
    
    stats = {
        'średnia': df[token_count_col].mean(),
        'mediana': df[token_count_col].median(),
        'min': df[token_count_col].min(),
        'max': df[token_count_col].max(),
        'suma': df[token_count_col].sum(),
        'odchylenie_std': df[token_count_col].std()
    }
    
    return stats

# padding tokenów do max_tokens podanych w parametrze

def pad_tokens(token_lists, pad_token_id=0, max_tokens=None):
    if max_tokens is None:
        max_tokens = max(len(tokens) for tokens in token_lists)
    padded_tokens = [
        tokens + [pad_token_id] * (max_tokens - len(tokens))
        for tokens in token_lists
    ]
    # zwróć jako numpy array, aby obiekty miały atrybut .shape i były kompatybilne z bibliotekami ML
    #return np.array(padded_tokens)
    return padded_tokens