import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import os

# Mueve todas tus funciones de ayuda a un solo lugar
from churn_library.helpers import (
    create_null_flags,
    remove_rows_with_any_null,
    clean_negatives,
    detect_and_convert_categoricals,
    create_new_features,
    sanitize_column_names)


#======PROCESSING TASKS=========
def remove_rows_with_any_null_task(input_path, output_path, config):
    """Airflow task: removes rows where specified subset columns are null."""
    print("--- Starting task: remove rows with any null ---")
    df = pd.read_csv(input_path, sep=';')

    # Leemos la lista de columnas desde la configuración
    subset_cols = config['preprocessing']['row_removal']['null_subset_cols']

    df_processed = remove_rows_with_any_null(df, subset_cols=subset_cols)

    print(f"Rows before: {len(df)}, Rows after: {len(df_processed)}")
    df_processed.to_csv(output_path, sep=';', index=False)
    print(f"Task finished. Data saved to {output_path}")


def clean_negatives_task(input_path, output_path, config):
    """Airflow task: removes rows with negative values in specified columns."""
    print("--- Starting task: clean negative values ---")
    df = pd.read_csv(input_path, sep=';')

    # Leemos la lista de columnas de excepción desde la configuración
    exception_cols = config['preprocessing']['negative_values']['exception_cols']

    df_processed = clean_negatives(df, exception_cols=exception_cols)

    df_processed.to_csv(output_path, sep=';', index=False)
    
    print(f"Task finished. Data saved to {output_path}")


def calculate_imputation_values_task(input_path, output_path_json, config):
    """
    Airflow task: reads the training data, calculates median values for specified
    columns, and saves them to a JSON file.
    This task "fits" our imputer.
    """
    print("--- Starting calculation of imputation values ---")
    df = pd.read_csv(input_path, sep=';')
    
    imputation_values = {}
    cols_for_median = config['preprocessing']['imputation']['median_fill_columns']
    
    print(f"Calculating medians for columns: {cols_for_median}")
    for col in cols_for_median:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            imputation_values[col] = median_val
            print(f"  - Median for '{col}': {median_val}")
        else:
            print(f"  - Warning: Column '{col}' not found or not numeric. Skipped.")
            
    print(f"Saving imputation values to {output_path_json}")
    with open(output_path_json, 'w') as f:
        json.dump(imputation_values, f, indent=4)
        
    print("Task finished.")



def create_null_flags_task(input_path, output_path, config):
    """Airflow task: reads data and creates binary flags for null values in specified columns."""
    print("--- Starting creation of null flags task ---")
    df = pd.read_csv(input_path, sep=';')
    
    cols_to_flag = config['preprocessing']['null_flags']['columns']
    
    # Llamamos a tu función de ayuda que ya existe
    df_processed = create_null_flags(df, subset_cols=cols_to_flag)
    
    df_processed.to_csv(output_path, sep=';', index=False)
    print(f"Task finished. Data with null flags saved to {output_path}")

def train_test_split_task(input_path, train_output_path, test_output_path, config):
    """
    Lee un archivo CSV, divide los datos en sets de entrenamiento y prueba,
    y los guarda en archivos CSV separados.
    """
    print(f"Leyendo datos desde: {input_path}")
    df = pd.read_csv(input_path, sep=';')

    target = config['target']

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['preprocessing']['imputation']['test_size'], random_state=config['random_state']
    )

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    print(f"Guardando set de entrenamiento en: {train_output_path}")
    df_train.to_csv(train_output_path, sep=';', index=False)
    
    print(f"Guardando set de prueba en: {test_output_path}")
    df_test.to_csv(test_output_path, sep=';', index=False)

    print("División de datos completada.")


def remove_columns_task(input_path, output_path, config):
    """
    Lee un archivo CSV, elimina las columnas especificadas y guarda el resultado.
    """
    print(f"Leyendo datos desde: {input_path}")
    df = pd.read_csv(input_path, sep=';')

    columns_to_drop = config['preprocessing']['column_removal']['irrelevant_columns']

    print(f"Eliminando {len(columns_to_drop)} columnas irrelevantes...")
    df_clean = df.drop(columns=columns_to_drop, errors='ignore')
    
    print(f"Columnas eliminadas. El nuevo DataFrame tiene {df_clean.shape[1]} columnas.")
    print(f"Guardando el resultado en: {output_path}")
    df_clean.to_csv(output_path, sep=';', index=False)

    return output_path

def impute_all_nulls_task(input_path, output_path, imputation_values_path, config):
    """
    Lee un CSV y realiza una imputación doble:
    1. Rellena columnas categóricas específicas con un valor constante (UNKN).
    2. Rellena columnas numéricas específicas con la mediana calculada del set de entrenamiento.
    """
    print("--- Iniciando la tarea de imputación consolidada ---")
    
    # Leer el DataFrame
    df = pd.read_csv(input_path, sep=';')
    
    # Cargar los valores de imputación (medianas) calculados previamente
    with open(imputation_values_path, 'r') as f:
        imputation_values_from_train = json.load(f)

    # 1. Imputar columnas categóricas con el valor constante 'UNKN'
    fill_value = config['preprocessing']['imputation']['categorical_fill_value']
    
    df = detect_and_convert_categoricals(df)

    # Identificar todas las columnas categóricas (object y category) en el DataFrame
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Columnas categóricas a imputar: {categorical_cols}")

    for col in categorical_cols:
        # --- Solución clave: Añadir la nueva categoría antes de rellenar ---
        # Si la columna es de tipo 'category', añadir la nueva categoría 'UNKN'
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].cat.add_categories([fill_value])
        
        # Ahora, rellenar los nulos
        df[col].fillna(fill_value, inplace=True)

    print(f"Imputación de TODAS las columnas categóricas completada con el valor: '{fill_value}'")
    # 2. Imputar columnas numéricas con la mediana
    columns_impute_median = config['preprocessing']['imputation']['median_fill_columns']

    for col in columns_impute_median:
        if col in imputation_values_from_train.keys():
            value = imputation_values_from_train[col]
            df[col].fillna(value, inplace=True)
        
    print(f"Imputación de columnas numéricas con la mediana completada.")
    
    # Verificar que no queden nulos después de la imputación
    print("\nVerificación final de nulos:")
    print(df[categorical_cols + columns_impute_median].isnull().sum())
    
    # Guardar el DataFrame final
    nulos_por_columna = df.isnull().sum()

    # Filtramos para mostrar solo las columnas que tienen al menos un valor nulo
    columnas_con_nulos = nulos_por_columna[nulos_por_columna > 0]
    print(f"NUMERO DE NULOS DESPUES DE PROCESAR {columnas_con_nulos}")
    df.to_csv(output_path, sep=';', index=False)
    print(f"\nTarea finalizada. Datos guardados en: {output_path}")

#======FEATURE ENGINEERING TASKS=========

def read_clean_data_task(input_path_train: str, input_path_test: str, output_path_train: str, output_path_test: str, config: dict):
    """
    Reads the train and test data sets, removes specified columns,
    and saves the results to separate CSV files.
    
    Args:
        input_path_train (str): Path to the train set CSV file.
        input_path_test (str): Path to the test set CSV file.
        output_path_train (str): Path to save the train set with removed columns.
        output_path_test (str): Path to save the test set with removed columns.
        config (dict): Configuration dictionary with the list of columns to remove.
    
    Returns:
        None
    """
    print("--- Starting read and column removal task ---")
    
    # 1. Read the train and test DataFrames
    try:
        df_train = pd.read_csv(input_path_train, sep=';')
        df_test = pd.read_csv(input_path_test, sep=';')
    except FileNotFoundError as e:
        print(f"Error: Could not find an input file. {e}")
        raise
        
    print(f"Data sets read. Train size: {df_train.shape}, Test size: {df_test.shape}")
    
    # 2. Get the list of columns to remove from the config file
    columns_to_drop = config['column_removal']['irrelevant_columns']
    print(f"Columns to drop: {columns_to_drop}")
    
    # 3. Drop the columns from both DataFrames
    # The 'errors='ignore'' parameter prevents the code from failing if a column doesn't exist
    df_train_processed = df_train.drop(columns=columns_to_drop, errors='ignore')
    df_test_processed = df_test.drop(columns=columns_to_drop, errors='ignore')
    
    print("Columns dropped.")
    print(f"New train size: {df_train_processed.shape}, New test size: {df_test_processed.shape}")
    
    # 4. Save the processed DataFrames to the output paths
    df_train_processed.to_csv(output_path_train, sep=';', index=False)
    df_test_processed.to_csv(output_path_test, sep=';', index=False)
    
    print(f"Train set saved to: {output_path_train}")
    print(f"Test set saved to: {output_path_test}")
    print("Task completed successfully.")


def create_new_features_task(input_path_train, input_path_test, output_path_train, output_path_test):
    """
    Reads the train and test sets, applies the feature engineering function,
    and saves the results.
    """
    print(f"Reading train set from: {input_path_train}")
    df_train = pd.read_csv(input_path_train, sep=';')
    
    print(f"Reading test set from: {input_path_test}")
    df_test = pd.read_csv(input_path_test, sep=';')
    
    print("Applying feature engineering to both sets...")
    df_train_new_features = create_new_features(df_train)
    df_test_new_features = create_new_features(df_test)
    
    print(f"Saving train set to: {output_path_train}")
    df_train_new_features.to_csv(output_path_train, sep=';', index=False)

    print(f"Saving test set to: {output_path_test}")
    df_test_new_features.to_csv(output_path_test, sep=';', index=False)
    
    print("Feature engineering task completed successfully.")


def binarize_train_set_task(input_path, output_path, columns, q: int, bins_path: str) -> pd.DataFrame:
    """
    Binariza las columnas del set de entrenamiento y guarda los límites de los bins.
    """
    print("Starting training set binning...")
    
    # Creamos una copia para evitar SettingWithCopyWarning
    df_train = pd.read_csv(input_path, sep=';')
    df_binned = df_train.copy()
    bins_map = {}
    
    for col in columns:
        # Usamos pd.qcut para obtener los bins y los límites
        df_binned[f'{col}_binned'], bins = pd.qcut(
            df_binned[col],
            q=q,
            labels=False, # Usamos etiquetas numéricas
            retbins=True,
            duplicates='drop'
        )
        bins_map[col] = bins
        
        # Elimina la columna original
        df_binned.drop(columns=[col], inplace=True)

    # Guardar el mapeo de los bins para usarlo en el set de prueba
    print(f"Saving bins in: {bins_path}")
    joblib.dump(bins_map, bins_path)
    
    print("Finished training set binning.")
    df_binned.to_csv(output_path, sep=';', index=False)

    return df_binned

# Nueva función para aplicar el binarizador en el set de prueba
def binarize_test_set_task(input_path, output_path, columns, bins_path: str) -> pd.DataFrame:
    """
    Aplica los límites de los bins guardados en el set de prueba.
    """

    print(f"Reading data from {input_path} and binarizing columns: {columns}")

    # Step 1: Read the data from the input path
    df_test = pd.read_csv(input_path, sep=';')

    print("Starting test set binning...")
    
    # Cargamos el mapeo de bins guardado
    print(f"Loading bins...: {bins_path}")
    bins_map = joblib.load(bins_path)

    df_binned = df_test.copy()
    
    for col in columns:
        bins = bins_map[col]
        df_binned[f'{col}_binned'] = pd.cut(
            df_binned[col],
            bins=bins,
            labels=False, 
            right=True,
            include_lowest=True
        )

        df_binned[f'{col}_binned'].fillna(-1, inplace=True)

        df_binned.drop(columns=[col], inplace=True)
        
    print("Test set binning completed.")
    df_binned.to_csv(output_path, sep=';', index=False)

    return  {'test_df_path': output_path}

def one_hot_encode_and_align_task(input_path_train, input_path_test, output_path_train, output_path_test):
    """
    Lee los sets de entrenamiento y prueba, detecta variables categóricas,
    aplica One-Hot Encoding y alinea sus columnas.
    """
    print("Starting the One-Hot Encoding and column alignment task...")
    

    df_train = pd.read_csv(input_path_train, sep=';')
    df_test = pd.read_csv(input_path_test, sep=';')

    df_train_clean = sanitize_column_names(df_train)
    df_test_clean = sanitize_column_names(df_test)

    df_train_converted = detect_and_convert_categoricals(df_train_clean)
    df_test_converted = detect_and_convert_categoricals(df_test_clean)

    categorical_cols_to_encode = df_train_converted.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Applying One-Hot Encoding to the following columns: {categorical_cols_to_encode}")

    df_train_encoded = pd.get_dummies(df_train_converted, columns=categorical_cols_to_encode, drop_first=True)
    
    df_test_encoded = pd.get_dummies(df_test_converted, columns=categorical_cols_to_encode, drop_first=True)
    
    final_train_cols = df_train_encoded.columns.tolist()
    df_test_aligned = df_test_encoded.reindex(columns=final_train_cols, fill_value=0)
    
    df_train_final = df_train_encoded.astype(int, errors='ignore')
    df_test_final = df_test_aligned.astype(int, errors='ignore')

    print(f"Number of columns in the train set: {df_train_encoded.shape[1]}")
    print(f"Number of columns in the aligned test set: {df_test_aligned.shape[1]}")

    # 8. Guarda los DataFrames procesados
    df_train_final.to_csv(output_path_train, sep=';', index=False)
    df_test_final.to_csv(output_path_test, sep=';', index=False)
    
    print("One-Hot Encoding and alignment task completed successfully.")

    return {
        'df_train_path': output_path_train,
        'df_test_path': output_path_test
    }


#===========TRAINING TASK

def train_model_task(input_path: str, model_config: dict, target_variable_name: str):
    """
    Lee datos, realiza validación cruzada si se especifica en la configuración, 
    entrena el modelo final con todo el set de entrenamiento y devuelve el modelo.

    Parámetros:
    - input_path (str): Ruta al archivo de datos de entrenamiento (train.csv).
    - config (dict): Diccionario de configuración con los parámetros del modelo.
    """
    print("Starting the model training task...")
    
    df = pd.read_csv(input_path, sep=';')

    X = df.drop(columns=[target_variable_name])
    y = df[target_variable_name]
    
    model_name = model_config.get('model_name')
    model_params = model_config.get('params', {})
    cv_folds = model_config.get('cv_folds')
    
    final_model = None

    if model_name == 'RandomForestClassifier':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=42, **model_params)
    elif model_name == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, **model_params)
    elif model_name == 'KNeighborsClassifier':
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(**model_params)
    elif model_name == 'LGBMClassifier':
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(random_state=42, **model_params)
    elif model_name == 'XGBClassifier':
        from xgboost import XGBClassifier
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **model_params)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # Escala los datos solo si el modelo lo requiere
    scaler = None
    if model_name in ['LogisticRegression', 'KNeighborsClassifier']:
        print("Scaling data with StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    if cv_folds and cv_folds > 1:
        print(f"Starting cross-validation with {cv_folds} folds for {model_name}...")
        scores = cross_val_score(model, X_scaled, y, cv=cv_folds, scoring='accuracy')
        avg_accuracy = scores.mean()
        print(f"\nCross-validation completed. Average Accuracy: {avg_accuracy:.4f}")
        print(f"Retraining the {model_name} on the full training dataset...")
        final_model = model
        final_model.fit(X_scaled, y)
        
    else:
        print(f"No cross-validation specified. Training {model_name} on the full dataset...")
        final_model = model
        final_model.fit(X_scaled, y)

    model_name_for_save = model_config.get('model_name')
    model_file_path = f"/opt/airflow/models/{model_name_for_save}_trained.pkl"
    scaler_file_path = f"/opt/airflow/models/{model_name_for_save}_scaler.pkl"

    # Guarda el modelo y el scaler
    print(f"Guardando el modelo en: {model_file_path}")
    joblib.dump(final_model, model_file_path)
    if scaler is not None:
        print(f"Guardando el scaler en: {scaler_file_path}")
        joblib.dump(scaler, scaler_file_path)

    return model_file_path
    

def evaluate_model_task(model_path: str, test_data_path: str, target_variable: str):
    """
    Tarea que lee un modelo y datos de prueba para evaluar su rendimiento.
    """
    print(f"Starting model evaluation from {test_data_path}...")
    
    try:
        # Carga el modelo
        model = joblib.load(model_path)
        
        # Carga el scaler si existe
        model_name = model_path.split('/')[-1].replace('_trained.pkl', '')
        scaler_path = f"/opt/airflow/models/{model_name}_scaler.pkl"
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        
        # Lee los datos de prueba
        df_test = pd.read_csv(test_data_path, sep=';')
        
    except FileNotFoundError:
        print("Error: Files not found. Ensure the training pipeline ran successfully.")
        return

    X_test = df_test.drop(columns=[target_variable])
    y_test = df_test[target_variable]
    
    # Escala los datos de prueba si el scaler existe
    if scaler is not None:
        print("Scaling test data with the saved StandardScaler...")
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("--------------------------------")

def predict_task(model_path: str, new_data_path: str, output_path: str, config):
    """
    Tarea que lee un modelo entrenado y genera predicciones sobre nuevos datos.

    Args:
        model_path (str): Ruta al modelo entrenado en formato .pkl.
        new_data_path (str): Ruta al archivo de datos nuevos a predecir.
        output_path (str): Ruta donde se guardarán las predicciones.
    """
    print(f"Starting prediction task...")

    try:
        # Lee el modelo ya entrenado
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Lee los datos nuevos (asume que están limpios y sanitizados)
        new_data_df = pd.read_csv(new_data_path, sep=';')
        
    except FileNotFoundError:
        print("Error: Files not found. Ensure model and data paths are correct.")
        return
    
    target = config['general']['target']
    if target in new_data_df.columns:
        print("Target variable found. Performing model evaluation...")
        X = new_data_df.drop(columns=[target])
        y_true = new_data_df[target]
        
        predictions = model.predict(X)
        
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        
        print("\n--- Model Evaluation Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print("--------------------------------")
        
        predictions_df = pd.DataFrame(new_data_df.copy())
        predictions_df['prediction'] = predictions
        
    else:
        print("Target variable not found. Generating predictions without evaluation.")
        predictions = model.predict(new_data_df)
        predictions_df = pd.DataFrame(new_data_df.copy())
        predictions_df['prediction'] = predictions

    # Guarda el DataFrame con las predicciones
    try:
        predictions_df.to_csv(output_path, sep=';', index=False)
        print(f"Predictions saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving predictions to {output_path}: {e}")
