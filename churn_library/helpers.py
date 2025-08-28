import re
import time
import pickle
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple

from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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

def clean_negatives(df, exception_cols):
    
    num_cols = df.select_dtypes(include='number').columns.tolist()

    columnas_finales_a_verificar = [
        col for col in num_cols if col not in exception_cols
    ]

    resumen, df_negativos = audit_negative_values(df, columnas_finales_a_verificar)

    if not df_negativos.empty:
        print("\n--- Cleaning negative values (if applicable) ---")
        
        # Obtenemos los índices de las filas problemáticas
        indices_a_eliminar = df_negativos.index
        
        # Eliminamos esas filas del DataFrame original
        df_limpio = df.drop(indices_a_eliminar)
        
        print(f"Original dataset length: {len(df)}")
        print(f"Length after cleaning negatives: {len(df_limpio)}")
        print(f"Deleted rows: {len(df) - len(df_limpio)}")
        return df_limpio
    else:
        print("\nDataset does not have anomalies in terms of negatives")
        df_limpio = df.copy()
        return df_limpio

def remove_rows_with_any_null(df, subset_cols):
    df_cleaned = df.dropna(subset=subset_cols, how='any')
    return df_cleaned

def create_null_flags(df, subset_cols):
    df_copy = df.copy()
    
    for col in subset_cols:
        if col in df_copy.columns:
            flag_col_name = f'{col}_isnull'
            df_copy[flag_col_name] = df_copy[col].isna().astype(int)
        else:
            print(f"Column '{col}' not found in DataFrame.")
            
    return df_copy


def impute_with_median(df, cols_to_fill):
    df_copy = df.copy()
    # ... (código de la función sin cambios)
    for col in cols_to_fill:
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            median_val = df_copy[col].median()
            df_copy[col] = df_copy[col].fillna(median_val)
    print(f"Imputed {len(cols_to_fill)} columns with their respective median.")
    return df_copy

# ---TASK FUNCTIONS

def find_columns_for_imputation(df, config_section):
    """
    Finds columns based on exact names, startswith patterns, and endswith patterns
    defined in a configuration dictionary.
    """
    cols_to_match = []

    exact_cols = config_section.get('exact_columns', [])
    for col in exact_cols:
        if col in df.columns and col not in cols_to_match:
            cols_to_match.append(col)

    patterns = config_section.get('patterns', {})
    if not patterns:
        return list(set(cols_to_match)) 

    for col in df.columns:
        if patterns.get('startswith'):
            for pattern in patterns['startswith']:
                if col.startswith(pattern) and col not in cols_to_match:
                    cols_to_match.append(col)
        if patterns.get('endswith'):
             for pattern in patterns['endswith']:
                if col.endswith(pattern) and col not in cols_to_match:
                    cols_to_match.append(col)

    return list(set(cols_to_match)) 

def audit_negative_values(df: pd.DataFrame, columnas: list):

    found_negatives = {}
    
    columns = [col for col in columnas if col in df.columns]
    
    for col in columns:
        num_negativos = (df[col] < 0).sum()
        if num_negativos > 0:
            found_negatives[col] = num_negativos

    if not found_negatives:
        return {}, pd.DataFrame()
    
    for col, count in found_negatives.items():
        print(f"- {col}: {count} negative values")
    
    mask_total_negatives = pd.Series([False] * len(df))
    cols_with_negatives = list(found_negatives.keys())
    
    for col in cols_with_negatives:
        mask_total_negatives = mask_total_negatives | (df[col].fillna(0) < 0)
        
    df_problematic = df[mask_total_negatives]
    
    print(df_problematic[cols_with_negatives])
    
    return found_negatives, df_problematic

def find_columns_by_pattern(df, patterns):
    cols_to_match = []
    for col in df.columns:
        if 'startswith' in patterns:
            for pattern in patterns['startswith']:
                if col.startswith(pattern):
                    cols_to_match.append(col)
                    break
        if 'endswith' in patterns and col not in cols_to_match:
             for pattern in patterns['endswith']:
                if col.endswith(pattern):
                    cols_to_match.append(col)
                    break
    return cols_to_match

def detect_and_convert_categoricals(df: pd.DataFrame, umbral_discreta: int = 20):
    """
    Detects numeric columns that are actually discrete/categorical
    and converts them to the 'category' type.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    umbral_discreta (int): The number of unique values below which
                           a numeric variable is considered categorical.
    
    Returns:
    tuple: A tuple with (the DataFrame with conversions, the list of converted columns).
    """
    df_copy = df.copy()
    variables_convertidas = []

    for col in df_copy.columns:
        if is_numeric_dtype(df_copy[col]) and col != 'churn':
            num_valores_unicos = df_copy[col].nunique()

            if num_valores_unicos <= umbral_discreta:
                df_copy[col] = df_copy[col].astype('category')
                variables_convertidas.append(col)
    
    print(f"Conversion completed. Converted {len(variables_convertidas)} columns to 'category' type.")
    print("Converted columns:", variables_convertidas)
    
    return df_copy

def group_category_by_churn(df: pd.DataFrame, column: str, upper_th: float, lower_th: float, new_column_name: str, target: str = 'churn'):
    df_copia = df.copy()

    churn_by_category = df_copia.groupby(column)[target].mean().reset_index()
    churn_by_category = churn_by_category.rename(columns={target: 'mean_churn'})

    # 2. Asignar la categoría de riesgo
    def asignar_riesgo(tasa):
        if tasa >= upper_th:
            return 'Alto Riesgo'
        elif tasa <= lower_th:
            return 'Bajo Riesgo'
        else:
            return 'Riesgo Moderado'

    churn_by_category[new_column_name] = churn_by_category['tasa_churn_media'].apply(asignar_riesgo)
    
    # 3. Crear el mapeo de categoría a grupo de riesgo
    mapeo = churn_by_category.set_index(column)[new_column_name]

    # 4. Unir la nueva columna de riesgo al DataFrame original
    df_final = pd.merge(df_copia, churn_by_category[[column, new_column_name]], on=column, how='left')

    return df_final, mapeo


def get_quantile_bins(df: pd.DataFrame, column: str, q: int = 5, labels: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Optional[List]]:
    """
    Bins a numeric column into quantiles, creating a new categorical column
     and returns the list of bins for use on a different dataset.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    column (str): The name of the numeric column to bin.
    q (int): The number of quantiles (bins) for the discretization.
    labels (List[str], optional): A list of labels for the bins.
                                    If None, default labels will be used.

    Returns:
    Tuple: A tuple containing:
           - pd.DataFrame: The DataFrame with the new binned column.
           - Optional[List]: The list of bins generated by pd.qcut.
    """
    if column not in df.columns:
        print(f"Error: Column '{column}' not found in the DataFrame.")
        return df, None
    if not is_numeric_dtype(df[column]):
        print(f"Error: Column '{column}' is not numeric.")
        return df, None

    df_copy = df.copy()
    binned_column_name = f'{column}_binned'

    if labels and len(labels) != q:
        print(f"Warning: The length of labels ({len(labels)}) does not match the number of quantiles ({q}). Default labels will be used.")
        labels = None
    
    try:
        # qcut returns the bins as a Categorical index
        binned_series, bins = pd.qcut(df_copy[column], q=q, labels=labels, duplicates='drop', retbins=True)
        df_copy[binned_series] = binned_series
        
        # Drop the original column to avoid multicollinearity
        df_copy.drop(columns=[column], inplace=True)
        
        print(f"Column '{column}' was binned into {q} ranges and saved in '{binned_column_name}'.")
        return df_copy, bins.tolist()
    
    except ValueError as e:
        print(f"Warning: Could not bin column '{column}': {e}")
        print(f"Column '{column}' will be kept as is.")
        return df, None

def binarize_and_align_quantiles(df_train: pd.DataFrame, df_test: pd.DataFrame, columns: List[str], q: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bins a set of columns in the train and test sets, using the quantiles
    from the train set to align the transformation on the test set.
    
    Args:
        df_train (pd.DataFrame): The training DataFrame.
        df_test (pd.DataFrame): The testing DataFrame.
        columns (List[str]): A list of numeric column names to bin.
        q (int): The number of quantiles for the discretization.
        
    Returns:
        Tuple: A tuple with (df_train_binned, df_test_binned).
    """
    df_train_binned = df_train.copy()
    df_test_binned = df_test.copy()
    
    for col in columns:
        print(f"--- Processing column '{col}' ---")
        
        # 1. First, create the bins and the labels from the training set
        #    We need to get the bins list to apply to the test set
        #    And we need to get the labels to apply to both sets
        df_train_binned[f'{col}_binned'], bins_list = pd.qcut(
            df_train_binned[col],
            q=q,
            labels=False, # Use integer labels
            retbins=True,
            duplicates='drop'
        )
        
        # 2. Apply the same bins and labels to the test set
        df_test_binned[f'{col}_binned'] = pd.cut(
            df_test_binned[col],
            bins=bins_list,
            labels=False, # Use integer labels
            right=True,
            include_lowest=True
        )
        
        # 3. Drop the original column to avoid data leakage
        df_train_binned.drop(columns=[col], inplace=True)
        df_test_binned.drop(columns=[col], inplace=True)
        
    print("\nBinning and alignment process completed.")
    return df_train_binned, df_test_binned
    
def create_new_features(df):
    df_copy = df.copy()

    #  Overage
    df_copy['total_overage_rev'] = df_copy['ovrrev_Mean'] + df_copy['vceovr_Mean'] + df_copy['datovr_Mean']
    df_copy['overage_ratio'] = np.where(
        df_copy['totmrc_Mean'] == 0,
        0,
        df_copy['total_overage_rev'] / df_copy['totmrc_Mean']
    )

    df_copy['complete_to_attempt_ratio'] = np.where(
        df_copy['attempt_Mean'] == 0,
        0, # Reemplazar con 0 si el denominador es 0
        df_copy['complete_Mean'] / df_copy['attempt_Mean']
    )

    # Uso y pago
    df_copy['mou_per_euro'] = np.where(
        df_copy['totmrc_Mean'] == 0,
        0, 
        df_copy['mou_Mean'] / df_copy['totmrc_Mean']
    )
    df_copy['relative_change_rev_mou'] = np.where(
        df_copy['change_mou'] == 0,
        0, 
        df_copy['change_rev'] / df_copy['change_mou']
    )
    df_copy['minutes_per_call'] = np.where(
        df_copy['comp_vce_Mean'] == 0,
        0, 
        df_copy['mou_cvce_Mean'] / df_copy['comp_vce_Mean']
    )

    df_copy['one_min_call_ratio'] = np.where(
        df_copy['recv_vce_Mean'] == 0,
        0, 
        df_copy['inonemin_Mean'] / df_copy['recv_vce_Mean']
    )
    variables_originales = [
        'ovrrev_Mean', 'vceovr_Mean', 'datovr_Mean', 'totmrc_Mean',
        'mou_cvce_Mean', 'mou_cdat_Mean', 'complete_Mean', 'attempt_Mean',
        'mou_Mean', 'change_rev', 'change_mou'
    ]
    df_final = df_copy.drop(columns=variables_originales, errors='ignore')
    
    return df_final