from pathlib import Path
from huggingface_hub import hf_hub_download

# /c:/GITHUB/ML_PROJEKT/tokenizer.py
# GitHub Copilot
# Krótki skrypt: preferuje js-tiktoken (lite) i domyślnie używa "o200K_base".
# Wymagania (opcjonalne): pip install js-tiktoken tiktoken huggingface_hub


def load_tiktoken(tokenizer_name: str = "o200K_base", repo_id: str = "openai/tiktoken", prefer_js: bool = True):
    """
    Próbuje w pierwszej kolejności użyć js-tiktoken (jeśli prefer_js=True).
    Jeśli to się nie uda, próbuje standardowego tiktoken.
    Jeżeli oba zawiodą, pobiera plik <tokenizer_name>.tiktoken z HF Hub i próbuje załadować.
    Zwraca obiekt z metodami encode(text) i decode(tokens).
    """
    # 1) Spróbuj js-tiktoken (lekka JS-implementacja) jeśli preferowane
    if prefer_js:
        try:
            import js_tiktoken  # pip install js-tiktoken
            # obsłuż kilka możliwych API
            if hasattr(js_tiktoken, "get_encoding"):
                return js_tiktoken.get_encoding(tokenizer_name)
            if hasattr(js_tiktoken, "encoding_for_model"):
                return js_tiktoken.encoding_for_model(tokenizer_name)
            # jeżeli js_tiktoken ma inną API, można tu dodać dodatkowe warunki
        except Exception:
            pass

    # 2) Spróbuj standardowego tiktoken (jeśli zainstalowany)
    try:
        import tiktoken
        return tiktoken.get_encoding(tokenizer_name)
    except Exception:
        pass

    # 3) Pobierz plik tokenizera z HF i spróbuj załadować plik
    tfname = f"{tokenizer_name}.tiktoken"
    try:
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=tfname)
    except Exception as e:
        raise RuntimeError(f"Nie udało się pobrać pliku tokenizera '{tfname}' z HF: {e}")

    # 3a) Spróbuj załadować przez standardowe tiktoken (Encoding.from_file)
    try:
        import tiktoken
        return tiktoken.Encoding.from_file(downloaded_path)
    except Exception:
        pass

    # 3b) Spróbuj załadować przez js_tiktoken jeśli dostępne i wspiera wczytywanie z pliku
    try:
        import js_tiktoken
        # możliwe warianty API; sprawdź dostępność i użyj jeśli istnieje
        if hasattr(js_tiktoken, "Tokenizer") and hasattr(js_tiktoken.Tokenizer, "from_file"):
            return js_tiktoken.Tokenizer.from_file(downloaded_path)
        if hasattr(js_tiktoken, "Encoding") and hasattr(js_tiktoken.Encoding, "from_file"):
            return js_tiktoken.Encoding.from_file(downloaded_path)
    except Exception:
        pass

    raise RuntimeError(f"Nie udało się załadować tokenizera '{tokenizer_name}' (brak kompatybilnego tiktoken/js_tiktoken). "
                       "Zainstaluj js-tiktoken lub tiktoken albo sprawdź nazwę tokenizera.")


if __name__ == "__main__":
    # Przykład użycia
    enc = load_tiktoken("o200K_base")  # domyślnie o200K_base, można zmienić
    text = "Witaj świecie! This is a test."
    tokens = enc.encode(text)
    decoded = enc.decode(tokens)

    print("Tekst:", text)
    print("Tokeny:", tokens)
    print("Odtworzony tekst:", decoded)