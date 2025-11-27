import pandas as pd
import numpy as np


def predict_top_classes(model, test_text, indeksy_model_1, 
                       column_name='NAZWA', top_n=5, model_type='gpt-3.5-turbo'):
    """
    Przewiduje top N najlepszych dopasowań klas dla podanego tekstu
    
    Args:
        model: wytrenowany model XGBoost
        test_text: tekst do klasyfikacji
        indeksy_model_1: DataFrame z danymi treningowymi (zawiera kolumnę Y)
        column_name: nazwa kolumny z tekstem (domyślnie 'NAZWA')
        top_n: liczba najlepszych przewidywań do zwrócenia (domyślnie 5)
        model_type: typ modelu tokenizacji (domyślnie 'gpt-3.5-turbo') 
    Returns:
        lista krotek (nazwa_klasy, prawdopodobieństwo)
    """
    from Funkcje.tokenizacja import tokenize_column, pad_tokens
    
    # Tokenizacja tekstu - tworzymy DataFrame z jednym wierszem
    test_df = pd.DataFrame({column_name: [test_text]})
    test_df_tokenized = tokenize_column(test_df, column_name=column_name, model=model_type)
    
    # Padding tokenów do tej samej długości co dane treningowe
    max_tokens = indeksy_model_1[f'{column_name}_TOKEN_COUNT'].max()
    test_tokens_padded = pad_tokens(
        test_df_tokenized[f'{column_name}_TOKENS'].tolist(), 
        pad_token_id=0, 
        max_tokens=max_tokens
    )
    
    # Przewidywanie
    pred_proba = model.predict_proba(test_tokens_padded)
    
    # Top N najlepszych dopasowań
    top_indices = np.argsort(pred_proba[0])[-top_n:][::-1]
    top_probs = pred_proba[0][top_indices]
    
    # Pobierz unikalne kategorie z danych treningowych
    unique_categories = sorted(indeksy_model_1['Y'].unique())
    
    # Mapowanie: model.classes_ zawiera liczby, mapujemy je na oryginalne nazwy
    top_classes = [
        unique_categories[idx] if idx < len(unique_categories) else f"Unknown_{idx}" 
        for idx in top_indices
    ]
    
    # Zwróć listę krotek (klasa, prawdopodobieństwo)
    return list(zip(top_classes, top_probs))



def print_predictions(predictions):
    """
    Wyświetla przewidywania w czytelnym formacie
    
    Args:
        predictions: lista krotek (nazwa_klasy, prawdopodobieństwo)
    """
    print(f"\nTop {len(predictions)} przewidywanych klas:")
    for i, (class_label, prob) in enumerate(predictions, 1):
        print(f"{i}. {class_label} (Prawdopodobieństwo: {prob:.4f})")


def predict_interactive(model, indeksy_model_1, top_n=5):
    """
    Interaktywna funkcja do przewidywania klas - pobiera tekst od użytkownika
    
    Args:
        model: wytrenowany model XGBoost
        indeksy_df_token: DataFrame z tokenizowanymi danymi treningowymi
        indeksy_model_1: DataFrame z danymi treningowymi
        top_n: liczba najlepszych przewidywań do zwrócenia
    """
    test_text = input("Wprowadź tutaj tekst do tokenizacji i przewidywania indeksu: ")
    
    predictions = predict_top_classes(
        model=model,
        test_text=test_text,
        indeksy_model_1=indeksy_model_1,
        top_n=top_n
    )
    
    print_predictions(predictions)
    
    return predictions

def predict_interactive_id_to_kod(model, indeksy_model_1, top_n=5, slowniki_df=None):
    """
    Interaktywna funkcja do przewidywania klas - pobiera tekst od użytkownika

    Args:
        model: wytrenowany model XGBoost
        indeksy_df_token: DataFrame z tokenizowanymi danymi treningowymi
        indeksy_model_1: DataFrame z danymi treningowymi
        top_n: liczba najlepszych przewidywań do zwrócenia
        slowniki_df: DataFrame z kolumnami SLIT_ID i KOD do mapowania
    """
    test_text = input("Wprowadź tutaj tekst do tokenizacji i przewidywania indeksu: ")

    predictions = predict_top_classes(
        model=model,
        test_text=test_text,
        indeksy_model_1=indeksy_model_1,
        top_n=top_n
    )
    # w predictions Mapowanie ID oddzielonych "_"na KOD na podstawie ramki danych slowniki_df. Ramka slowniki_df zawiera kolumny slit_id i kod
    if slowniki_df is not None:
        mapped_predictions = []
        for class_label, prob in predictions:
            # Sprawdź, czy klasa zawiera "_"
            if "_" in str(class_label):
                slit_ids = str(class_label).split("_")
                kod_parts = []

                # Mapuj każdy SLIT_ID na odpowiadający KOD
                for slit_id_str in slit_ids:
                    try:
                        # Konwersja na int dla kompatybilności z różnymi typami danych
                        slit_id = int(slit_id_str)
                        # Znajdź odpowiadający KOD w slowniki_df
                        kod_row = slowniki_df[slowniki_df['SLIT_ID'] == slit_id]
                        if not kod_row.empty:
                            kod = kod_row['KOD'].values[0]
                            kod_parts.append(str(kod))
                        else:
                            kod_parts.append(slit_id_str)  # Jeśli nie znaleziono, zachowaj oryginalny ID
                    except (ValueError, TypeError):
                        kod_parts.append(slit_id_str)  # Jeśli konwersja się nie powiodła, zachowaj oryginalny string

                # Połącz KOD-y za pomocą "-"
                combined_kod = "-".join(kod_parts)
                mapped_predictions.append((combined_kod, prob))
            else:
                mapped_predictions.append((class_label, prob))  # Jeśli nie ma "_", zachowaj oryginalną etykietę
        predictions = mapped_predictions



    print_predictions(predictions)

    return predictions
