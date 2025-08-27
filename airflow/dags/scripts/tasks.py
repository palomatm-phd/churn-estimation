import json
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
from churn_library.processing_helpers import (
    find_columns_for_imputation, 
    impute_with_constant, 
    create_null_flags,
    remove_rows_with_any_null,
    clean_negatives)

from churn_library.feature_helpers import (create_new_features, 
                                           binarize_and_align_quantiles,
                                           detect_and_convert_categoricals)

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


def impute_zeros_task(input_path, output_path, config):
    """Airflow task: reads CSV, imputes with 0 on specified columns, and writes CSV."""
    print("--- Starting zero imputation task ---")
    df = pd.read_csv(input_path, sep=';')

    # --- LÍNEAS DE DEPURACIÓN ---
    print("\n--- DEBUGGING: Column Search ---")
    print(f"Total columns available in DataFrame ({len(df.columns)}):")
    print(df.columns.tolist())
    
    imputation_config = config.get('preprocessing', {}).get('imputation', {})
    
    zero_fill_config = {
        'patterns': imputation_config.get('zero_fill_patterns', {}),
        'exact_columns': imputation_config.get('zero_fill_exact_columns', [])
    }
    print(f"\nConfiguration being used for search: {zero_fill_config}")
    print("--- END DEBUGGING ---\n")
    # ------------------------------------

    # Usamos la función inteligente de la librería para encontrar las columnas
    cols_for_zero = find_columns_for_imputation(df, zero_fill_config)

    print(f"Found {len(cols_for_zero)} columns to impute with zero: {cols_for_zero}")

    df_processed = impute_with_constant(df, cols_for_zero, fill_value=0)
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


def apply_imputation_task(input_path_data, output_path_data, input_path_json):
    """
    Airflow task: reads a dataset and the saved imputation values from a JSON file,
    then applies the imputation.
    This task "transforms" our data.
    """
    print(f"--- Applying imputation from {input_path_json} to {input_path_data} ---")
    df = pd.read_csv(input_path_data, sep=';')
    
    print("Loading imputation values...")
    with open(input_path_json, 'r') as f:
        imputation_values = json.load(f)
        
    print(f"Applying median imputation for columns: {list(imputation_values.keys())}")
    # Usamos .fillna directamente con el diccionario, es más eficiente
    df.fillna(value=imputation_values, inplace=True)
            
    df.to_csv(output_path_data, sep=';', index=False)
    print(f"Task finished. Transformed data saved to {output_path_data}")


def impute_categoricals_task(input_path, output_path, config):
    """Airflow task: reads CSV, imputes categorical columns with 'UNKN', and writes CSV."""
    print("--- Starting categorical imputation task ---")
    df = pd.read_csv(input_path, sep=';')
    fill_value = config['preprocessing']['imputation']['categorical_fill_value']
    cols_for_unkn = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # We use your helper function to impute
    df_processed = impute_with_constant(df, cols_for_unkn, fill_value=fill_value)
    df_processed.to_csv(output_path, sep=';', index=False)
    print(f"Task finished. Data saved to {output_path}")


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

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    print(f"Guardando set de entrenamiento en: {train_output_path}")
    train_df.to_csv(train_output_path, sep=';', index=False)
    
    print(f"Guardando set de prueba en: {test_output_path}")
    test_df.to_csv(test_output_path, sep=';', index=False)

    print("División de datos completada.")

def impute_with_unkn_task(input_path, output_path, config):
    """Airflow task: reads CSV, imputes categorical columns with 'UNKN', and writes CSV."""
    print("--- Starting categorical imputation task ---")
    df = pd.read_csv(input_path, sep=';')
    fill_value = config['preprocessing']['imputation']['categorical_fill_value']
    cols_for_unkn = config['preprocessing']['imputation']['unkn_fill_exact_columns']
    # We use your helper function to impute
    df_processed = impute_with_constant(df, cols_for_unkn, fill_value=fill_value)
    df_processed.to_csv(output_path, sep=';', index=False)
    print(f"Task finished. Data saved to {output_path}")


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
    cols_for_unkn = config['preprocessing']['imputation']['unkn_fill_exact_columns']
    
    for col in cols_for_unkn:
        df[col].fillna(fill_value, inplace=True)
        
    print(f"Imputación de columnas categóricas completada con el valor: '{fill_value}'")

    # 2. Imputar columnas numéricas con la mediana
    columns_impute_median = config['preprocessing']['imputation']['median_fill_columns']

    for col in columns_impute_median:
        if col in imputation_values_from_train.keys():
            value = imputation_values_from_train[col]
            df[col].fillna(value, inplace=True)
        
    print(f"Imputación de columnas numéricas con la mediana completada.")
    
    # Verificar que no queden nulos después de la imputación
    print("\nVerificación final de nulos:")
    print(df[cols_for_unkn + columns_impute_median].isnull().sum())
    
    # Guardar el DataFrame final
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


def one_hot_encode_and_align_task(input_path_train, input_path_test, output_path_train, output_path_test, config):
    """
    Reads the train and test sets, applies One-Hot Encoding, and aligns their columns.
    """
    print("Starting the One-Hot Encoding and column alignment task...")
    
    # Read the DataFrames
    df_train = pd.read_csv(input_path_train, sep=';')
    df_test = pd.read_csv(input_path_test, sep=';')
    
    # Get the list of categorical variables to encode
    categorical_cols_to_encode = config['preprocessing']['categorical_for_ohe'] # Assumed this list is in the config

    # Perform One-Hot Encoding on the train set
    df_train_encoded = pd.get_dummies(df_train, columns=categorical_cols_to_encode, drop_first=True)
    
    # Save the list of ALL resulting columns
    final_train_cols = df_train_encoded.columns.tolist()

    # Perform One-Hot Encoding on the test set
    df_test_encoded = pd.get_dummies(df_test, columns=categorical_cols_to_encode, drop_first=True)
    
    # Align the columns of the test set with the train set
    df_test_aligned = df_test_encoded.reindex(columns=final_train_cols, fill_value=0)
    
    print(f"Number of columns in the train set: {df_train_encoded.shape[1]}")
    print(f"Number of columns in the aligned test set: {df_test_aligned.shape[1]}")

    # Save the processed DataFrames
    df_train_encoded.to_csv(output_path_train, sep=';', index=False)
    df_test_aligned.to_csv(output_path_test, sep=';', index=False)

    print("One-Hot Encoding and alignment task completed successfully.")


def binarize_and_align_quantiles_task(input_path_train: str, input_path_test: str, output_path_train: str, output_path_test: str, columns_to_bin: List[str], q: int):
    """
    Lee los sets de entrenamiento y prueba, binariza y alinea sus columnas.
    """
    print(f"Leyendo set de entrenamiento desde: {input_path_train}")
    df_train = pd.read_csv(input_path_train, sep=';')
    
    print(f"Leyendo set de prueba desde: {input_path_test}")
    df_test = pd.read_csv(input_path_test, sep=';')
    
    # Llama a tu función principal para binarizar y alinear
    df_train_binned, df_test_binned = binarize_and_align_quantiles(df_train, df_test, columns_to_bin, q)
    
    # Guardar los DataFrames procesados
    df_train_binned.to_csv(output_path_train, sep=';', index=False)
    df_test_binned.to_csv(output_path_test, sep=';', index=False)
    
    print("Tarea de binarización y alineación completada con éxito.")

def one_hot_encode_and_align_task(input_path_train, input_path_test, output_path_train, output_path_test):
    """
    Reads the train and test sets, automatically detects categorical variables,
    applies One-Hot Encoding, and aligns their columns.
    """
    print("Starting the One-Hot Encoding and column alignment task...")
    
    # Read the DataFrames
    df_train = pd.read_csv(input_path_train, sep=';')
    df_test = pd.read_csv(input_path_test, sep=';')
    
    # Detect and convert categorical columns on the train set
    df_train_converted, converted_cols = detect_and_convert_categoricals(df_train)
    
    # Apply the same conversion to the test set using the list from the train set
    df_test_converted, _ = detect_and_convert_categoricals(df_test)
    
    # Get the list of ALL categorical variables (both object and converted)
    categorical_cols_to_encode = df_train_converted.select_dtypes(include=['object', 'category']).columns.tolist()

    print(f"Applying One-Hot Encoding to the following columns: {categorical_cols_to_encode}")

    # Perform One-Hot Encoding on the train set
    df_train_encoded = pd.get_dummies(df_train_converted, columns=categorical_cols_to_encode, drop_first=True)
    
    # Perform One-Hot Encoding on the test set
    df_test_encoded = pd.get_dummies(df_test_converted, columns=categorical_cols_to_encode, drop_first=True)
    
    # Save the list of ALL resulting columns from the training set
    final_train_cols = df_train_encoded.columns.tolist()

    # Align the columns of the test set with the train set, filling missing columns with 0
    df_test_aligned = df_test_encoded.reindex(columns=final_train_cols, fill_value=0)
    
    print(f"Number of columns in the train set: {df_train_encoded.shape[1]}")
    print(f"Number of columns in the aligned test set: {df_test_aligned.shape[1]}")

    # Save the processed DataFrames
    df_train_encoded.to_csv(output_path_train, sep=';', index=False)
    df_test_aligned.to_csv(output_path_test, sep=';', index=False)

    print("One-Hot Encoding and alignment task completed successfully.")