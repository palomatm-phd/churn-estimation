import json
import pandas as pd
from churn_library.processing_helpers import (
    find_columns_for_imputation, 
    impute_with_constant, 
    create_null_flags,
    remove_rows_with_any_null,
    clean_negatives)


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
    
    # CORRECCIÓN IMPORTANTE: Accedemos a la configuración anidada correctamente
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