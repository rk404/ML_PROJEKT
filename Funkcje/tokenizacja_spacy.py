import spacy
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Załaduj model spaCy dla języka polskiego
# Użyj: python -m spacy download pl_core_news_sm aby pobrać model
try:
    nlp = spacy.load('pl_core_news_sm')
except OSError:
    print('Model pl_core_news_sm nie jest zainstalowany.')
    print('Zainstaluj model poleceniem: python -m spacy download pl_core_news_sm')
    # Fallback na model angielski
    try:
        nlp = spacy.load('en_core_web_sm')
        print('Używam modelu angielskiego jako zastępczego')
    except OSError:
        print('Brak dostępnych modeli spaCy. Zainstaluj model.')
        nlp = None


def spacy_tokenizer(text):
    """
    Tokenizer spaCy dla TfidfVectorizer
    
    Args:
        text: tekst do tokenizacji
    
    Returns:
        lista tokenów
    """
    if nlp is None:
        return text.split()
    
    doc = nlp(str(text))
    return [token.text for token in doc if not token.is_punct]


def spacy_lemmatizer(text):
    """
    Lematyzator spaCy dla TfidfVectorizer
    
    Args:
        text: tekst do lematyzacji
    
    Returns:
        lista lematów
    """
    if nlp is None:
        return text.split()
    
    doc = nlp(str(text))
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]


def tokenize_column_spacy_tfidf(df, column_name='NAZWA', lemmatize=True, remove_stopwords=True, 
                                max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95,
                                return_vectorizer=False):
    """
    Tokenizuje tekst w wybranej kolumnie DataFrame używając spaCy + TF-IDF
    
    Args:
        df: DataFrame z danymi
        column_name: nazwa kolumny do tokenizacji (domyślnie 'NAZWA')
        lemmatize: czy używać lematyzacji (domyślnie True)
        remove_stopwords: czy usuwać stop words (domyślnie True)
        max_features: maksymalna liczba cech TF-IDF (domyślnie 5000)
        ngram_range: zakres n-gramów (domyślnie (1, 2) = unigramy i bigramy)
        min_df: minimalna częstość dokumentów (domyślnie 2)
        max_df: maksymalna częstość dokumentów jako % (domyślnie 0.95)
        return_vectorizer: czy zwrócić również obiekt vectorizer
    
    Returns:
        DataFrame z dodatkowymi kolumnami lub (DataFrame, vectorizer, tfidf_matrix) jeśli return_vectorizer=True
    """
    if nlp is None:
        raise ValueError('Model spaCy nie jest załadowany. Zainstaluj model spaCy.')
    
    # Utwórz kopię DataFrame
    df_copy = df.copy()
    
    # Przygotuj dane tekstowe
    texts = df_copy[column_name].fillna('').astype(str).values
    
    # Wybierz tokenizer
    tokenizer = spacy_lemmatizer if lemmatize else spacy_tokenizer
    
    # Utwórz TfidfVectorizer
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        lowercase=True
    )
    
    # Dopasuj i transformuj
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Funkcja tokenizująca pojedynczy tekst (dla kolumny TOKENS)
    def tokenize_text(text):
        if pd.isna(text):
            return []
        doc = nlp(str(text))
        
        if lemmatize:
            tokens = [token.lemma_ for token in doc if not token.is_punct]
            if remove_stopwords:
                tokens = [t for t in tokens if not any(nlp(t)[0].is_stop for _ in [0])]
        else:
            tokens = [token.text for token in doc if not token.is_punct]
            if remove_stopwords:
                tokens = [t for t, tok in zip([token.text for token in doc], doc) if not tok.is_punct and not tok.is_stop]
        
        return tokens
    
    # Tokenizuj kolumnę
    df_copy[f'{column_name}_TOKENS'] = df_copy[column_name].apply(tokenize_text)
    
    # Dodaj kolumnę z liczbą tokenów
    df_copy[f'{column_name}_TOKEN_COUNT'] = df_copy[f'{column_name}_TOKENS'].apply(len)
    
    # Dodaj kolumnę z wektorami TF-IDF (sparse matrix row)
    df_copy[f'{column_name}_TFIDF'] = [tfidf_matrix[i].toarray().flatten() for i in range(tfidf_matrix.shape[0])]
    
    print(f'Tokenizacja zakończona.')
    print(f'Średnia liczba tokenów: {df_copy[f"{column_name}_TOKEN_COUNT"].mean():.2f}')
    print(f'Maksymalna liczba tokenów: {df_copy[f"{column_name}_TOKEN_COUNT"].max()}')
    print(f'Minimalna liczba tokenów: {df_copy[f"{column_name}_TOKEN_COUNT"].min()}')
    print(f'Rozmiar słownika TF-IDF: {len(vectorizer.get_feature_names_out())}')
    print(f'Kształt macierzy TF-IDF: {tfidf_matrix.shape}')
    
    if return_vectorizer:
        return df_copy, vectorizer, tfidf_matrix
    return df_copy


def get_top_tfidf_terms(tfidf_matrix, vectorizer, top_n=10, doc_index=0):
    """
    Pobiera top N terminów z najwyższym TF-IDF dla dokumentu
    
    Args:
        tfidf_matrix: macierz TF-IDF
        vectorizer: obiekt TfidfVectorizer
        top_n: liczba terminów do zwrócenia
        doc_index: indeks dokumentu
    
    Returns:
        lista krotek (termin, wartość TF-IDF)
    """
    feature_names = vectorizer.get_feature_names_out()
    doc_vector = tfidf_matrix[doc_index].toarray().flatten()
    
    # Sortuj indeksy według wartości TF-IDF
    top_indices = doc_vector.argsort()[-top_n:][::-1]
    
    return [(feature_names[i], doc_vector[i]) for i in top_indices if doc_vector[i] > 0]


def transform_new_texts(texts, vectorizer):
    """
    Transformuje nowe teksty używając istniejącego vectorizera
    
    Args:
        texts: lista tekstów do transformacji
        vectorizer: wytrenowany obiekt TfidfVectorizer
    
    Returns:
        macierz TF-IDF dla nowych tekstów
    """
    if isinstance(texts, pd.Series):
        texts = texts.fillna('').astype(str).values
    
    return vectorizer.transform(texts)


def get_token_statistics(df, column_name='NAZWA'):
    """
    Zwraca statystyki tokenizacji dla kolumny
    
    Args:
        df: DataFrame z tokenizowanymi danymi
        column_name: nazwa bazowej kolumny
    
    Returns:
        słownik ze statystykami
    """
    token_count_col = f'{column_name}_TOKEN_COUNT'
    
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


def extract_pos_tags(df, column_name='NAZWA'):
    """
    Wyodrębnia części mowy (POS tags) z tekstu
    
    Args:
        df: DataFrame z danymi
        column_name: nazwa kolumny do analizy
    
    Returns:
        DataFrame z dodatkowymi kolumnami zawierającymi POS tags
    """
    if nlp is None:
        raise ValueError('Model spaCy nie jest załadowany.')
    
    df_copy = df.copy()
    
    def get_pos_tags(text):
        if pd.isna(text):
            return []
        doc = nlp(str(text))
        return [(token.text, token.pos_) for token in doc]
    
    df_copy[f'{column_name}_POS'] = df_copy[column_name].apply(get_pos_tags)
    
    return df_copy


def extract_named_entities(df, column_name='NAZWA'):
    """
    Wyodrębnia nazwane encje (NER) z tekstu
    
    Args:
        df: DataFrame z danymi
        column_name: nazwa kolumny do analizy
    
    Returns:
        DataFrame z dodatkowymi kolumnami zawierającymi encje
    """
    if nlp is None:
        raise ValueError('Model spaCy nie jest załadowany.')
    
    df_copy = df.copy()
    
    def get_entities(text):
        if pd.isna(text):
            return []
        doc = nlp(str(text))
        return [(ent.text, ent.label_) for ent in doc.ents]
    
    df_copy[f'{column_name}_ENTITIES'] = df_copy[column_name].apply(get_entities)
    
    return df_copy


def tokens_to_text(tokens):
    """
    Łączy tokeny z powrotem w tekst
    
    Args:
        tokens: lista tokenów
    
    Returns:
        połączony tekst
    """
    return ' '.join(tokens)
