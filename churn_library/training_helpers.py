import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import re

def train_and_evaluate_model(model_name, hyperparameters, X_train, y_train, X_test, y_test):
    """
    Performs cross-validation, trains, and evaluates a classification model.
    (Version 2.0 with Cross-Validation and Logistic Regression)
    """
    
    print("\n" + "="*60)
    print(f"--- PROCESSING MODEL: {model_name} ---")
    print("="*60)
    
    # 1. MODEL SELECTION
    if model_name == 'RandomForest':
        model = RandomForestClassifier(**hyperparameters)
    elif model_name == 'XGBoost':
        model = XGBClassifier(**hyperparameters)
    elif model_name == 'LightGBM':
        model = LGBMClassifier(**hyperparameters)
    elif model_name == 'LogisticRegression':
        hyperparameters.setdefault('max_iter', 2000)
        model = LogisticRegression(**hyperparameters)
    else:
        raise ValueError("Unrecognized model.")

    # --- 2. CROSS-VALIDATION (ON TRAIN SET) ---
    print(f"\nRunning Cross-Validation for {model_name} (this may take a while)...")
    start_time = time.time()
    
    # We use cv=5 for a robust evaluation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
    
    cv_duration = time.time() - start_time
    print(f"Cross-Validation completed in {cv_duration:.2f} seconds.")
    
    # --- 3. FINAL TRAINING (ON THE ENTIRE TRAIN SET) ---
    print(f"\nTraining {model_name} with all training data...")
    model.fit(X_train, y_train)
    print("Final model trained!")

    # --- 4. PREDICTION AND EVALUATION (ON THE TEST SET) ---
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc_test = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'], output_dict=True)
    f1_churn = report['Churn']['f1-score']
    
    print(f"\n--- Mean AUC in Cross-Validation: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f}) ---")
    print(f"--- AUC on Test Set: {auc_test:.4f} ---")
    
    # --- 5. CONFUSION MATRIX ---
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Churn', 'Churn'])
    disp.plot(cmap='plasma')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()
    
    # --- 6. RETURN RESULTS ---
    results = {
        'model_name': model_name,
        'AUC_CV_Mean': np.mean(cv_scores),
        'AUC_Test': auc_test,
        'F1-Score_Churn': f1_churn
    }
    
    return model, results

def model_evaluation(models, X_train, y_train, X_test, y_test):
    results = []
    names = []
    trained_models = {}

    print("--- Evaluación de modelos sin Cross-Validation ---")
    for name, model in models:
        print(f"\nProcesando modelo: {name}...")

        # Se maneja el escalado de datos solo para modelos que lo necesitan
        if isinstance(model, LogisticRegression) or isinstance(model, SVC) or isinstance(model, KNeighborsClassifier):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train_to_fit = X_train_scaled
            X_test_to_predict = X_test_scaled
        else:
            X_train_to_fit = X_train
            X_test_to_predict = X_test

        # Entrenar el modelo
        start_time = time.time()
        model.fit(X_train_to_fit, y_train)
        fit_duration = time.time() - start_time
        print(f"Modelo {name} entrenado en {fit_duration:.2f} segundos.")

        # Hacer predicciones
        y_pred = model.predict(X_test_to_predict)
        y_pred_proba = model.predict_proba(X_test_to_predict)[:, 1]

        # Calcular métricas y almacenar resultados
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)

        results.append({'model_name': name, 'accuracy': accuracy, 'auc': auc_score})
        names.append(name)

        msg = f"Modelo {name}: Accuracy = {accuracy:.4f}, AUC = {auc_score:.4f}"
        print(msg)
        
        # --- Almacenar el modelo entrenado y guardarlo en un archivo ---
        trained_models[name] = model
        filename = f'{name}_model.pckl'
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Modelo guardado en '{filename}'.")

    # Opcional: imprimir todos los resultados al final
    print("\n--- Resumen de Resultados ---")
    results_df = pd.DataFrame(results, columns=['model_name', 'accuracy', 'auc'])
    print(results_df)

    return trained_models

def sanitize_column_names(df: pd.DataFrame):
    """
    Sanitizes DataFrame column names to be compatible with LightGBM/XGBoost,
    by removing special characters.
    
    Args:
        df (pd.DataFrame): The DataFrame with columns that need cleaning.
        
    Returns:
        pd.DataFrame: The DataFrame with cleaned column names.
    """
    df_copy = df.copy()
    new_columns = []
    for col in df_copy.columns:
        # Replace special characters and spaces with an underscore
        new_col = re.sub(r'[\[\]<>]', '', col)
        new_col = new_col.replace(' ', '_')
        new_col = re.sub(r'[^A-Za-z0-9_]+', '', new_col)
        new_columns.append(new_col)
    
    df_copy.columns = new_columns
    return df_copy