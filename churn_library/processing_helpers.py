import pandas as pd

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