from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split    
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np


def train_xgb_model(indeksy_model, n_estimators, n_learning_rate, n_max_depth, n_test_size=0.2, n_random_state=42):
    """
    Trenuje model XGBoost do klasyfikacji indeksów.
    
    Parametry:
    -----------
    indeksy_model_1 : DataFrame
        DataFrame zawierający kolumny 'NAZWA_TOKENS_PADDED' i 'Indeks_2_bloki'
    n_estimators : int, default=100
        Liczba drzew w modelu
    learning_rate : float, default=0.1
        Współczynnik uczenia
    max_depth : int, default=5
        Maksymalna głębokość drzewa
    test_size : float, default=0.2
        Proporcja zbioru testowego
    random_state : int, default=42
        Ziarno dla generatora liczb losowych
    
    Zwraca:
    (model, y_test, y_pred, accuracy_score, classification_report, confusion_matrix) - wytrenowany model i metryki
    """
    #max_tokens = indeksy_model['NAZWA_TOKENS_PADDED'].apply(len).max()
    # x na podstawie kolumny 'NAZWA_TOKENS_PADDED'
    X = np.array(indeksy_model['NAZWA_TOKENS_PADDED'].tolist())
    le = LabelEncoder()
    y = le.fit_transform(indeksy_model['Y'])
   # y = indeksy_model['Y'].astype('category').cat.codes  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=n_test_size, random_state=n_random_state)
    
    model = XGBClassifier(n_estimators=n_estimators, learning_rate=n_learning_rate, max_depth=n_max_depth, random_state=n_random_state)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    #metryki accuracy_score, classification_report, confusion_matrix

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)
    
    return model, y_test, y_pred, accuracy, report, matrix

def train_xgb_model_Seg3(data, n_estimators, n_learning_rate, n_max_depth, n_test_size, n_random_state):
    X = np.array(data['NAZWA_TOKENS_PADDED'].tolist())
    # scalar columns as 2D columns (convert missing values appropriately)
    slit1 = data['SLIT_ID_1'].astype(float).to_numpy().reshape(-1, 1)
    slit2 = data['SLIT_ID_2'].astype(float).to_numpy().reshape(-1, 1)
    # final feature matrix: [slit1, slit2, token_features...]
    X = np.hstack([slit1, slit2, X])  
    y = data['Y'].astype(str).values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=n_test_size, random_state=n_random_state)

    model = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=n_learning_rate,
        max_depth=n_max_depth,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    y_test_decoded = le.inverse_transform(y_test)
    y_pred_decoded = le.inverse_transform(y_pred)

    # Generujemy raport na podstawie stringów - bez podawania labels/target_names ręcznie
    report = classification_report(y_test_decoded, y_pred_decoded, zero_division=0)
    matrix = confusion_matrix(y_test_decoded, y_pred_decoded)

    return model, y_test, y_pred, accuracy, report, matrix