import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import math
from pandas.api.types import is_numeric_dtype
import seaborn as sns

def calcular_nulos(df):
    """
    Calcula el número de valores nulos y su porcentaje en cada columna del DataFrame.
    """
    null_counts = df.isnull().sum()
    null_percentages = (df.isnull().sum() / len(df)) * 100

    null_info = pd.DataFrame({
        'Nulos': null_counts,
        'Porcentaje': null_percentages
    })

    return null_info

def crear_flag_nulos(df: pd.DataFrame, columna_a_verificar: str, nueva_columna_flag: str) -> pd.DataFrame:
    """
    Crea una columna binaria (flag) en un DataFrame que indica la presencia de valores nulos
    en una columna específica.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        columna_a_verificar (str): El nombre de la columna donde se buscarán los valores nulos.
        nueva_columna_flag (str): El nombre de la nueva columna que contendrá el flag (1 si es nulo, 0 si no lo es).

    Returns:
        pd.DataFrame: El DataFrame original con la nueva columna flag añadida.
    """
    if columna_a_verificar not in df.columns:
        raise ValueError(f"La columna '{columna_a_verificar}' no se encuentra en el DataFrame.")

    boolean_mask = df[columna_a_verificar].isnull()
    df[nueva_columna_flag] = np.where(boolean_mask, 1, 0)

    print(f"Número de 1s en la nueva columna flag: {df[df[columna_a_verificar].isnull()][nueva_columna_flag].value_counts()}\n")
    
    return df

# Esto también podría sernos útil para observar las distribuciones de otras variables
def graficar_distribuciones(df: pd.DataFrame, columnas: list, n_cols: int = 3, bins: int = 30):
    """
    Genera y muestra histogramas para una lista de columnas de un DataFrame,
    ignorando los valores nulos.

    La función organiza los gráficos en una cuadrícula cuyo número de columnas se puede definir.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        columnas (list): Una lista de strings con los nombres de las columnas a graficar.
        n_cols (int): El número de columnas que tendrá la cuadrícula de gráficos. Por defecto es 3.
        bins (int): El número de contenedores (bins) para cada histograma. Por defecto es 30.
    """
    # Validar que las columnas existan en el DataFrame
    for col in columnas:
        if col not in df.columns:
            print(f"Advertencia: La columna '{col}' no se encuentra en el DataFrame y será ignorada.")
            columnas.remove(col)
            
    if not columnas:
        print("No hay columnas válidas para graficar.")
        return

    # Calcular el número de filas necesarias para la cuadrícula
    n_rows = math.ceil(len(columnas) / n_cols)

    # Crear la figura y los subplots (ejes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    
    # Aplanar el array de ejes para iterar fácilmente, incluso si es de una sola fila/columna
    axes = axes.flatten()

    # Iterar sobre cada columna y su eje correspondiente para crear el gráfico
    for i, col_nombre in enumerate(columnas):
        # Seleccionar los datos no nulos
        datos_no_nulos = df[col_nombre].dropna()
        
        # Graficar el histograma en el eje correspondiente
        axes[i].hist(datos_no_nulos, bins=bins, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribución de {col_nombre}')
        axes[i].set_xlabel(col_nombre)
        axes[i].set_ylabel('Frecuencia')

        min_val = datos_no_nulos.min()
        max_val = datos_no_nulos.max()
        axes[i].set_xlim(left=min_val, right=max_val)
    # Ocultar los ejes sobrantes si el número de gráficos no completa la cuadrícula
    for i in range(len(columnas), len(axes)):
        axes[i].set_visible(False)

    # Ajustar el diseño para que no se superpongan los títulos y etiquetas
    plt.tight_layout()
    plt.show()

def imputar_valores_nulos(df: pd.DataFrame, columnas: list, estrategia: str = 'mediana', valor_relleno: str = 'UNKN') -> pd.DataFrame:
    """
    Imputa valores nulos en columnas de un DataFrame usando una estrategia definida.

    Soporta estrategias numéricas ('media', 'mediana', 'moda') y una estrategia
    para valores constantes ('constante'), útil para variables categóricas.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        columnas (list): Lista de nombres de las columnas a imputar.
        estrategia (str): Método de imputación. Acepta 'mediana', 'media', 'moda', o 'constante'.
                          Por defecto es 'mediana'.
        valor_relleno (str): El valor a usar cuando la estrategia es 'constante'.
                             Por defecto es 'UNKN'.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con los valores nulos imputados.
    """
    df_imputado = df.copy()
    
    estrategias_validas = ['mediana', 'media', 'moda', 'constante']
    if estrategia.lower() not in estrategias_validas:
        raise ValueError(f"Estrategia no válida. Elija entre: {estrategias_validas}")

    print(f"--- Iniciando imputación con la estrategia: '{estrategia}' ---")

    for col in columnas:
        if col not in df_imputado.columns:
            print(f"Advertencia: La columna '{col}' no se encuentra en el DataFrame y será ignorada.")
            continue

        if df_imputado[col].isnull().any():
            valor_imputacion = None
            
            # --- Lógica para estrategias numéricas ---
            if estrategia in ['mediana', 'media', 'moda']:
                # Verificar que la columna sea numérica para estas estrategias
                if not is_numeric_dtype(df_imputado[col]):
                    print(f"Advertencia: La columna '{col}' no es numérica. La estrategia '{estrategia}' no se puede aplicar. Se omite la columna.")
                    continue
                
                if estrategia == 'mediana':
                    valor_imputacion = df_imputado[col].median()
                elif estrategia == 'media':
                    valor_imputacion = df_imputado[col].mean()
                elif estrategia == 'moda':
                    valor_imputacion = df_imputado[col].mode()[0]
                
                print(f"-> Columna numérica '{col}' imputada con la {estrategia}: {valor_imputacion:.2f}")

            # --- Lógica para la estrategia de constante (categórica o numérica) ---
            elif estrategia == 'constante':
                valor_imputacion = valor_relleno
                print(f"-> Columna '{col}' imputada con el valor constante: '{valor_imputacion}'")

            # Aplicar la imputación
            df_imputado[col] = df_imputado[col].fillna(valor_imputacion)
            
        else:
            print(f"-> Columna '{col}' no tiene valores nulos. No se requiere imputación.")
            
    return df_imputado

def graficar_distribuciones_categoricas(df: pd.DataFrame, columnas: list, n_cols: int = 3):
    """
    Genera y muestra gráficos de barras para visualizar la distribución de
    variables categóricas en un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame que contiene los datos.
        columnas (list): Una lista de strings con los nombres de las columnas categóricas a graficar.
        n_cols (int): El número de columnas que tendrá la cuadrícula de gráficos. Por defecto es 3.
    """
    # Validar que las columnas existan en el DataFrame
    columnas_validas = [col for col in columnas if col in df.columns]
    if not columnas_validas:
        print("Ninguna de las columnas proporcionadas se encuentra en el DataFrame.")
        return

    # Calcular el número de filas necesarias
    n_rows = math.ceil(len(columnas_validas) / n_cols)

    # Crear la figura y los subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5))
    axes = axes.flatten()

    # Iterar sobre cada columna para crear su gráfico de barras
    for i, col_nombre in enumerate(columnas_validas):
        ax = axes[i]
        # Usamos sns.countplot, la herramienta perfecta para esta tarea
        sns.countplot(x=col_nombre, data=df, ax=ax, palette='viridis', hue=col_nombre, legend=False)
        ax.set_title(f'Distribución de {col_nombre}')
        ax.set_xlabel('Categoría')
        ax.set_ylabel('Frecuencia (Conteo)')

        # Opcional: Para mejorar la legibilidad de los ejes con 0 y 1
        # Si la columna es de tipo numérico (int/float) con valores 0 y 1, cambiamos las etiquetas
        if pd.api.types.is_numeric_dtype(df[col_nombre]) and set(df[col_nombre].unique()).issubset({0, 1}):
            ax.set_xticklabels(['No (0)', 'Sí (1)'])
            ax.set_xlabel('Respuesta')

    # Ocultar los ejes sobrantes
    for i in range(len(columnas_validas), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()

def analizar_churn_categorica(df: pd.DataFrame, columna: str, target: str = 'churn'):
    """
    Versión mejorada que analiza y visualiza la tasa de churn para una variable categórica
    con dos gráficos: distribución y tasa de churn.
    """
    if columna not in df.columns:
        print(f"Error: La columna '{columna}' no se encuentra en el DataFrame.")
        return

    print(f"--- Análisis de Churn para la Variable Categórica: '{columna}' ---")

    tasa_churn_general = df[target].mean()
    analisis = df.groupby(columna)[target].agg(['mean', 'count']).rename(
        columns={'mean': 'Tasa de Churn', 'count': 'Total Clientes'}
    ).sort_values(by='Tasa de Churn', ascending=False)
    
    print(analisis)
    print("\n")

    # --- Creación de la visualización con dos subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'Análisis de Churn por "{columna}"', fontsize=18, weight='bold')

    # Gráfico 1: Distribución de Clientes
    sns.barplot(x=analisis.index, y=analisis['Total Clientes'], ax=axes[0], palette='viridis', hue=analisis.index)
    axes[0].set_title('Distribución de Clientes por Categoría', fontsize=14)
    axes[0].set_xlabel(columna, fontsize=12)
    axes[0].set_ylabel('Número de Clientes', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    # Gráfico 2: Tasa de Churn
    sns.barplot(x=analisis.index, y=analisis['Tasa de Churn'], ax=axes[1], palette='plasma', hue=analisis.index)
    axes[1].axhline(tasa_churn_general, color='red', linestyle='--', 
                    label=f'Tasa General ({tasa_churn_general:.2%})')
    axes[1].set_title('Tasa de Churn por Categoría', fontsize=14)
    axes[1].set_xlabel(columna, fontsize=12)
    axes[1].set_ylabel('Tasa de Churn', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajustar para el título principal
    plt.show()

def analizar_churn_numerica(df: pd.DataFrame, columna: str, target: str = 'churn', q: int = 5):
    """
    Versión mejorada que analiza y visualiza la tasa de churn para una variable numérica
    con dos gráficos: distribución por churn y tasa de churn por rangos.
    """
    if columna not in df.columns:
        print(f"Error: La columna '{columna}' no se encuentra en el DataFrame.")
        return
    if not is_numeric_dtype(df[columna]):
        print(f"Error: La columna '{columna}' no es numérica.")
        return

    print(f"--- Análisis de Churn para la Variable Numérica: '{columna}' ---")
    
    df_copy = df.copy()
    
    # --- Creación de la visualización con dos subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'Análisis de Churn por "{columna}"', fontsize=18, weight='bold')

    # Gráfico 1: Distribución de la variable por valor de Churn
    sns.histplot(data=df_copy, x=columna, hue=target, multiple="dodge", kde=True, ax=axes[0], palette='coolwarm')
    axes[0].set_title(f'Distribución de "{columna}" por Churn', fontsize=14)
    axes[0].set_xlabel(columna, fontsize=12)
    axes[0].set_ylabel('Frecuencia', fontsize=12)
    
    # Gráfico 2: Tasa de Churn por rangos de la variable
    columna_binned = f'{columna}_rango'
    try:
        df_copy[columna_binned] = pd.qcut(df_copy[columna], q=q, duplicates='drop')
        
        tasa_churn_general = df_copy[target].mean()
        analisis_binned = df_copy.groupby(columna_binned)[target].agg(['mean', 'count']).rename(
            columns={'mean': 'Tasa de Churn', 'count': 'Total Clientes'}
        )
        
        print("Análisis por rangos:")
        print(analisis_binned)
        print("\n")

        sns.barplot(x=analisis_binned.index, y=analisis_binned['Tasa de Churn'], ax=axes[1], palette='rocket', hue=analisis_binned.index)
        axes[1].axhline(tasa_churn_general, color='blue', linestyle='--', 
                        label=f'Tasa General ({tasa_churn_general:.2%})')
        axes[1].set_title(f'Tasa de Churn por Rangos de "{columna}"', fontsize=14)
        axes[1].set_xlabel(f'Rangos de {columna} ({q} cuantiles)', fontsize=12)
        axes[1].set_ylabel('Tasa de Churn', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend()

    except ValueError as e:
        axes[1].text(0.5, 0.5, f"No se pudo generar el gráfico de rangos:\n{e}", 
                     ha='center', va='center', transform=axes[1].transAxes)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def auditar_valores_negativos(df: pd.DataFrame, columnas: list):
    """
    Audita un DataFrame para encontrar valores negativos en un conjunto de columnas.

    Esta función revisa las columnas especificadas, imprime un resumen de los
    valores negativos encontrados y muestra las filas problemáticas.

    Args:
        df (pd.DataFrame): El DataFrame que se va a analizar.
        columnas (list): Una lista de nombres de las columnas que se deben verificar.

    Returns:
        tuple: Una tupla que contiene:
            - dict: Un diccionario con las columnas que tienen negativos y su conteo.
            - pd.DataFrame: Un DataFrame que contiene las filas completas con valores negativos.
              Estará vacío si no se encuentran negativos.
    """
    negativos_encontrados = {}
    
    # --- PASO 1: Contar los valores negativos en cada columna ---
    columnas_existentes = [col for col in columnas if col in df.columns]
    
    for col in columnas_existentes:
        num_negativos = (df[col] < 0).sum()
        if num_negativos > 0:
            negativos_encontrados[col] = num_negativos

    # --- PASO 2: Mostrar un resumen de los hallazgos ---
    print("--- Auditoría de Valores Negativos ---")
    if not negativos_encontrados:
        print("¡Buenas noticias! No se encontraron valores negativos en las columnas especificadas.")
        # Si no hay negativos, devolvemos un diccionario y un DataFrame vacíos
        return {}, pd.DataFrame()
    
    print("Se encontraron valores negativos en las siguientes columnas:")
    for col, count in negativos_encontrados.items():
        print(f"- {col}: {count} valores negativos")
    
    # --- PASO 3: Aislar y mostrar las filas problemáticas ---
    print("\n--- Mostrando filas con al menos un valor negativo detectado ---")
    
    mascara_negativos_total = pd.Series([False] * len(df))
    columnas_con_negativos = list(negativos_encontrados.keys())
    
    for col in columnas_con_negativos:
        # Usamos .fillna(0) para que los NaNs no den problemas en la comparación
        mascara_negativos_total = mascara_negativos_total | (df[col].fillna(0) < 0)
        
    df_problematicos = df[mascara_negativos_total]
    
    # Mostrar solo las columnas que tenían negativos para una vista más limpia
    print(df_problematicos[columnas_con_negativos])
    
    # --- PASO 4: Devolver los resultados para uso posterior ---
    return negativos_encontrados, df_problematicos

# --- Función para graficar la tasa de churn de una variable CATEGÓRICA ---
def plot_tasa_churn_categorica(df, columna, ax, target='churn'):
    """Dibuja un barplot de la tasa de churn para una variable categórica en un subplot (ax) específico."""
    tasa_churn_general = df[target].mean()
    
    analisis = df.groupby(columna)[target].agg(['mean', 'count']).rename(
        columns={'mean': 'Tasa de Churn', 'count': 'Total Clientes'}
    ).sort_values(by='Tasa de Churn', ascending=False)
    
    sns.barplot(x=analisis.index, y=analisis['Tasa de Churn'], ax=ax, palette='viridis', hue=analisis.index)
    ax.axhline(tasa_churn_general, color='red', linestyle='--', label=f'Tasa General ({tasa_churn_general:.2%})')
    ax.set_title(f'Churn por "{columna}"', fontsize=12)
    ax.set_xlabel('') # Quitamos la etiqueta X para no sobrecargar
    ax.set_ylabel('Tasa de Churn', fontsize=10)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.legend(fontsize=8)

# --- Función para graficar la tasa de churn de una variable NUMÉRICA ---
def plot_tasa_churn_numerica(df, columna, ax, target='churn', q=5):
    """Dibuja un barplot de la tasa de churn para una variable numérica (bineada) en un subplot (ax) específico."""
    df_copy = df.copy()
    columna_binned = f'{columna}_rango'
    
    try:
        df_copy[columna_binned] = pd.qcut(df_copy[columna], q=q, duplicates='drop')
        
        tasa_churn_general = df_copy[target].mean()
        analisis_binned = df_copy.groupby(columna_binned)[target].agg(['mean', 'count']).rename(
            columns={'mean': 'Tasa de Churn', 'count': 'Total Clientes'}
        )
        
        sns.barplot(x=analisis_binned.index.astype(str), y=analisis_binned['Tasa de Churn'], ax=ax, palette='plasma', hue=analisis_binned.index.astype(str))
        ax.axhline(tasa_churn_general, color='blue', linestyle='--', label=f'Tasa General ({tasa_churn_general:.2%})')
        ax.set_title(f'Churn por "{columna}"', fontsize=12)
        ax.set_xlabel('') # Quitamos la etiqueta X para no sobrecargar
        ax.set_ylabel('Tasa de Churn', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.legend(fontsize=8)
        
    except ValueError as e:
        ax.text(0.5, 0.5, f"No se pudo binnear '{columna}'\n{e}", ha='center', va='center')
