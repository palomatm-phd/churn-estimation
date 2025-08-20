import pandas as pd

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