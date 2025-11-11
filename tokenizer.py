import tiktoken
def load_tiktoken(model_name="o200K_base"):
    '''
    Ładuje tokenizer tiktoken dla podanego modelu
    Domyślnie używa "o200K_base"
    '''
    try:
        enc = tiktoken.get_encoding(model_name)
        print(f'Załadowano tokenizer dla modelu: {model_name}')
        return enc
    except Exception as e:
        print(f'Błąd ładowania tokenizera dla modelu {model_name}: {e}')
        return None 
