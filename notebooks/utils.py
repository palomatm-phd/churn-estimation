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
    Versión mejorada que analiza y visualiza el total de churn para una variable categórica.
    """
    if columna not in df.columns:
        print(f"Error: La columna '{columna}' no se encuentra en el DataFrame.")
        return

    print(f"--- Análisis de Churn para la Variable Categórica: '{columna}' ---")

    # Agregando el total de churn a la tabla de análisis
    analisis = df.groupby(columna).agg(
        {'churn': ['mean', 'count', 'sum']}
    ).rename(
        columns={'mean': 'Tasa de Churn', 'count': 'Total Clientes', 'sum': 'Total Churn'}
    )
    analisis.columns = analisis.columns.droplevel(0)
    analisis = analisis.sort_values(by='Total Churn', ascending=False)
    
    print(analisis)
    print("\n")

    # --- Creación de la visualización con un solo plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle(f'Total de Clientes con Churn por "{columna}"', fontsize=16, weight='bold')

    # Gráfico que muestra el número absoluto de clientes con churn
    sns.barplot(x=analisis.index, y=analisis['Total Churn'], ax=ax, palette='plasma', hue=analisis.index)
    ax.set_title(f'Número Absoluto de Clientes con Churn por Categoría', fontsize=14)
    ax.set_xlabel(columna, fontsize=12)
    ax.set_ylabel('Número Total de Clientes con Churn', fontsize=12)
    ax.tick_params(axis='x', rotation=45)

    # Añadir las etiquetas de valor en las barras para mayor claridad
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def analizar_churn_numerica(df: pd.DataFrame, columna: str, target: str = 'churn', q: int = 5, umbral_discreta: int = 20):
    """
    Versión mejorada que analiza y visualiza el total de churn para una variable numérica.
    Detecta automáticamente si la variable es discreta o continua. Para las discretas,
    las ordena por valor.
    """
    if columna not in df.columns:
        print(f"Error: La columna '{columna}' no se encuentra en el DataFrame.")
        return
    if not is_numeric_dtype(df[columna]):
        print(f"Error: La columna '{columna}' no es numérica.")
        return

    print(f"--- Análisis de Churn para la Variable Numérica: '{columna}' ---")
    
    df_copy = df.copy()
    
    # Lógica para determinar si la variable es discreta o continua
    if df[columna].nunique() <= umbral_discreta:
        print(f"La columna '{columna}' es tratada como discreta (<= {umbral_discreta} valores únicos).")
        
        # Agregando el total de churn a la tabla de análisis
        analisis = df_copy.groupby(columna).agg(
            {'churn': ['mean', 'count', 'sum']}
        ).rename(
            columns={'mean': 'Tasa de Churn', 'count': 'Total Clientes', 'sum': 'Total Churn'}
        )
        analisis.columns = analisis.columns.droplevel(0)
        
        # Ordenar por el valor de la variable, no por la cantidad de churn
        analisis.sort_index(inplace=True)
        
        print(analisis)
        print("\n")
        
        # --- Creación del gráfico ---
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.suptitle(f'Total de Clientes con Churn por "{columna}" (Valores Discretos)', fontsize=16, weight='bold')

        sns.barplot(x=analisis.index.astype(str), y=analisis['Total Churn'], ax=ax, palette='plasma', hue=analisis.index.astype(str))
        ax.set_title(f'Número Absoluto de Clientes con Churn por "{columna}"', fontsize=14)
        ax.set_xlabel(columna, fontsize=12)
        ax.set_ylabel('Número Total de Clientes con Churn', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.0f', fontsize=10)
    
    else:
        print(f"La columna '{columna}' es tratada como continua (> {umbral_discreta} valores únicos).")
        
        # --- Creación del gráfico ---
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.suptitle(f'Total de Clientes con Churn por "{columna}" (Rangos Cuantiles)', fontsize=16, weight='bold')

        columna_binned = f'{columna}_rango'
        try:
            df_copy[columna_binned] = pd.qcut(df_copy[columna], q=q, duplicates='drop')
            
            # Agregando el total de churn a la tabla de análisis
            analisis_binned = df_copy.groupby(columna_binned).agg(
                {'churn': ['mean', 'count', 'sum']}
            ).rename(
                columns={'mean': 'Tasa de Churn', 'count': 'Total Clientes', 'sum': 'Total Churn'}
            )
            analisis_binned.columns = analisis_binned.columns.droplevel(0)
            
            # Ordenar por el valor del bin, no por la cantidad de churn
            # La línea de sort_values ha sido eliminada.
            
            print("Análisis por rangos:")
            print(analisis_binned)
            print("\n")

            sns.barplot(x=analisis_binned.index.astype(str), y=analisis_binned['Total Churn'], ax=ax, palette='rocket', hue=analisis_binned.index.astype(str))
            ax.set_title(f'Número Absoluto de Clientes con Churn por Rangos de "{columna}"', fontsize=14)
            ax.set_xlabel(f'Rangos de {columna} ({q} cuantiles)', fontsize=12)
            ax.set_ylabel('Número Total de Clientes con Churn', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.0f', fontsize=10)

        except ValueError as e:
            ax.text(0.5, 0.5, f"No se pudo generar el gráfico de rangos:\n{e}", 
                     ha='center', va='center', transform=ax.transAxes)
            
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

