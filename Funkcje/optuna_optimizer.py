import optuna
from Funkcje.model import train_xgb_model


def optimize_xgb_model(indeksy_model, n_trials=20, n_estimators_range=(50, 300), 
                       learning_rate_range=(0.01, 0.3), max_depth_range=(3, 10),
                       test_size=0.2, random_state=42):
    """
    Optymalizuje hiperparametry modelu XGBoost używając Optuna.
    
    Parametry:
    -----------
    indeksy_model : DataFrame
        DataFrame z danymi do trenowania
    n_trials : int, default=20
        Liczba prób optymalizacji
    n_estimators_range : tuple, default=(50, 300)
        Zakres wartości dla n_estimators (min, max)
    learning_rate_range : tuple, default=(0.01, 0.3)
        Zakres wartości dla learning_rate (min, max)
    max_depth_range : tuple, default=(3, 10)
        Zakres wartości dla max_depth (min, max)
    test_size : float, default=0.2
        Proporcja zbioru testowego
    random_state : int, default=42
        Ziarno dla generatora liczb losowych
    
    Zwraca:
    --------
    dict : Słownik z najlepszymi parametrami i dokładnością
        {
            'best_params': dict z najlepszymi hiperparametrami,
            'best_accuracy': float z najlepszą dokładnością,
            'study': obiekt Optuna study
        }
    """
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', n_estimators_range[0], n_estimators_range[1])
        learning_rate = trial.suggest_float('learning_rate', learning_rate_range[0], learning_rate_range[1])
        max_depth = trial.suggest_int('max_depth', max_depth_range[0], max_depth_range[1])
        
        model, y_test, y_pred, accuracy, report, matrix = train_xgb_model(
            indeksy_model,
            n_estimators=n_estimators,
            n_learning_rate=learning_rate,
            n_max_depth=max_depth,
            n_test_size=test_size,
            n_random_state=random_state
        )
        
        return accuracy
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    # Wyświetl wyniki
    print('Best trial:')
    trial = study.best_trial
    print(f'  Accuracy: {trial.value:.4f}')
    print('  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
    
    return {
        'best_params': trial.params,
        'best_accuracy': trial.value,
        'study': study
    }