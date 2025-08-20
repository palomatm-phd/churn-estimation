# scripts/1_data_processing.py

import pandas as pd
from scripts import utils
import json

def clean_negatives(df, exception_cols = ['change_rev', 'change_mou']):
    
    num_cols = df.select_dtypes(include='number').columns.tolist()

    columnas_finales_a_verificar = [
        col for col in num_cols if col not in exception_cols
    ]

    resumen, df_negativos = utils.auditar_valores_negativos(df, columnas_finales_a_verificar)

    if not df_negativos.empty:
        print("\n--- Cleaning negative values (if applicable) ---")
        
        # Obtenemos los índices de las filas problemáticas
        indices_a_eliminar = df_negativos.index
        
        # Eliminamos esas filas del DataFrame original
        df_limpio = df.drop(indices_a_eliminar)
        
        print(f"Original dataset length: {len(df)}")
        print(f"Length after cleaning negatives: {len(df_limpio)}")
        print(f"Deleted rows: {len(df) - len(df_limpio)}")
    else:
        print("\nDataset does not have anomalies in terms of negatives")
        df_limpio = df.copy() # Creamos una copia para seguir trabajando

def remove_rows_with_any_null(df, subset_cols=['dualband', 'area']):
    df_cleaned = df.dropna(subset=subset_cols, how='any')
    
    return df_cleaned

#['infobase', 'rev_Mean', 'change_rev', 'avg6qty', 'hnd_price', 'prizm_social_one', 'hnd_webcap', 'ownrent', 'dwlltype', 'HHstatin', 'dwllsize', 'lor', 'adults', 'income', 'numbcars']
def create_null_flags(df, subset_cols):
    df_copy = df.copy()
    
    for col in subset_cols:
        if col in df_copy.columns:
            flag_col_name = f'{col}_isnull'
            df_copy[flag_col_name] = df_copy[col].isna().astype(int)
        else:
            print(f"Column '{col}' not found in DataFrame.")
            
    return df_copy

#['_Mean', 'change_', 'avg6'] with 0
#['hnd_price', 'lor', 'adults', 'income', 'numbcars'] with median
def impute_with_constant(df, cols_to_fill, fill_value):
    df_copy = df.copy()
    df_copy[cols_to_fill] = df_copy[cols_to_fill].fillna(fill_value)
    print(f"Imputed {len(cols_to_fill)} columns with constant value: '{fill_value}'")
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

def find_columns_by_pattern(df, patterns):
    """Helper function to find columns that match specified patterns."""
    cols_to_match = []
    for col in df.columns:
        if patterns.get('startswith'):
            for pattern in patterns['startswith']:
                if col.startswith(pattern) and col not in cols_to_match:
                    cols_to_match.append(col)
        if patterns.get('endswith'):
             for pattern in patterns['endswith']:
                if col.endswith(pattern) and col not in cols_to_match:
                    cols_to_match.append(col)
    return cols_to_match

def impute_zeros_task(input_path, output_path, config):
    """Airflow task: reads CSV, imputes with 0 according to patterns, and writes CSV."""
    print("--- Starting zero imputation task ---")
    df = pd.read_csv(input_path, sep=';')
    patterns = config['zero_fill_patterns']
    # We use the helper function to find the columns
    cols_for_zero = find_columns_by_pattern(df, patterns)
    # We use your helper function to impute
    df_processed = impute_with_constant(df, cols_for_zero, fill_value=0)
    df_processed.to_csv(output_path, sep=';', index=False)
    print(f"Task finished. Data saved to {output_path}")

#def impute_medians_task(input_path, output_path, config):
#    """Airflow task: reads CSV, imputes with median on specific columns, and writes CSV."""
#    print("--- Starting median imputation task ---")
#    df = pd.read_csv(input_path, sep=';')
#    cols_for_median = config['median_fill_columns']
#    # We use your helper function to impute
#    df_processed = impute_with_median(df, cols_for_median)
#    df_processed.to_csv(output_path, sep=';', index=False)
#    print(f"Task finished. Data saved to {output_path}")

def calculate_imputation_values_task(input_path, output_path_json, config):
    """
    Airflow task: reads the training data, calculates median values for specified
    columns, and saves them to a JSON file.
    This task "fits" our imputer.
    """
    print("--- Starting calculation of imputation values ---")
    df = pd.read_csv(input_path, sep=';')
    
    imputation_values = {}
    cols_for_median = config['median_fill_columns']
    
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
    fill_value = config['categorical_fill_value']
    cols_for_unkn = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # We use your helper function to impute
    df_processed = impute_with_constant(df, cols_for_unkn, fill_value=fill_value)
    df_processed.to_csv(output_path, sep=';', index=False)
    print(f"Task finished. Data saved to {output_path}")

def create_null_flags_task(input_path, output_path, config):
    """Airflow task: reads data and creates binary flags for null values in specified columns."""
    print("--- Starting creation of null flags task ---")
    df = pd.read_csv(input_path, sep=';')
    
    # Usamos la configuración de la variable de Airflow para saber en qué columnas crear flags.
    # Podemos usar las mismas que luego imputaremos.
    cols_to_flag = config['null_flag_columns'] 
    
    # Llamamos a tu función de ayuda que ya existe
    df_processed = create_null_flags(df, subset_cols=cols_to_flag)
    
    df_processed.to_csv(output_path, sep=';', index=False)
    print(f"Task finished. Data with null flags saved to {output_path}")