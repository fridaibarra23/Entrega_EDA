

#%%
def relacion_nulos_target(df):
    # Crear un diccionario para almacenar los resultados
    lista = {}
    
    # Filtrar columnas que tienen valores nulos
    columnas_con_nulos = df.columns[df.isnull().any()]
    
    # Calcular el porcentaje de nulos por TARGET para cada columna
    for col in columnas_con_nulos:
        porcentajes_nulos = df.groupby('TARGET')[col].apply(lambda x: (x.isnull().sum() / len(x)) * 100)
        lista[col] = porcentajes_nulos  # Asignar al diccionario

    return lista

#%%
def categorizar_columnas(df):
    # Columnas booleanas puras (tipo bool) y tipo int con solo 1 y 0
    col_bool = [col for col in df.columns if df[col].dtype == 'bool' or
                     (df[col].dtype == 'int64' and set(df[col].dropna().unique()) <= {0, 1})]
    
    # Columnas numéricas (int y float) que no son booleanas en disguise
    col_num = [col for col in df.select_dtypes(include=['int64', 'float']).columns 
                    if col not in col_bool]
    
    # Columnas categóricas (objetos y categorías)
    col_cat = df.select_dtypes(include=['object', 'category']).columns.tolist()

    return col_bool, col_cat, col_num


#%%
# Códigos de color ANSI
YELLOW = '\033[93m'  # Numeric
BLUE = '\033[94m'    # Categoric
PINK = '\033[38;5;205m'  # Boolean
RESET = '\033[0m'    # Reset color

def data_summary(df):
    for col in df.columns:
        # Detectar si la columna tiene solo valores 0 y 1 (tratarlas como booleanas)
        if df[col].isin([0, 1]).all():
            column_type = f"{PINK}boolean{RESET}"
        elif df[col].dtype in ['int64', 'float64']:  # Identificar columnas numéricas
            column_type = f"{YELLOW}numeric{RESET}"
        elif df[col].dtype == 'bool':  # Identificar columnas booleanas
            column_type = f"{PINK}boolean{RESET}"
        elif df[col].dtype == 'object':  # Identificar columnas categóricas
            column_type = f"{BLUE}categoric{RESET}"
        else:
            column_type = df[col].dtype  # Para otros tipos, sin color

        # Nombre de la columna y tipo de dato con color
        print(f"{col} ({column_type}) :", end=" ")
        
        # Tipo de dato detallado sin color
        print(f"(Type: {df[col].dtype})", end=" ")

        # Mostrar valores según el tipo de columna
        if column_type == f"{PINK}boolean{RESET}":
            unique_values = df[col].unique()
            if len(unique_values) == 1:  # Si la columna solo tiene un valor único
                print(f"Unique: [{unique_values[0]}]")
            else:
                print(f"Unique: {list(unique_values)}")
        
        elif column_type == f"{YELLOW}numeric{RESET}":
            min_val = df[col].min()
            max_val = df[col].max()
            mean_val = df[col].mean()
            print(f"Range = [{min_val:.2f} to {max_val:.2f}], Mean = {mean_val:.2f}")
        
        elif column_type == f"{BLUE}categoric{RESET}":
            unique_values = df[col].unique()
            # Mostrar los primeros 5 valores únicos
            print(f"Values: {unique_values[:5]}{' ...' if len(unique_values) > 5 else ''}")

        print()  # Línea en blanco para separar columnas


# %%

def cramers_v(confusion_matrix):
    """ 
    Calculate Cramer's V statistic for categorical-categorical association.
    Uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    """
    chi2 = scipy.stats.chi2_contingency(confusion_matrix)[0]  # Cambio aquí para usar scipy.stats directamente
    n = confusion_matrix.sum().sum()  # Total de observaciones
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


# %%

def get_deviation_of_mean_perc(df, list_var_continuous, target, multiplier):
    """
    Devuelve un DataFrame que muestra el porcentaje de valores que exceden el intervalo de confianza,
    junto con la distribución del target para esos valores.
    
    :param df: DataFrame con los datos a analizar.
    :param list_var_continuous: Lista de variables continuas (e.g., columnas_numericas_sin_booleanas).
    :param target: Variable objetivo para analizar la distribución de categorías.
    :param multiplier: Factor multiplicativo para determinar el rango de confianza (media ± multiplier * std).
    :return: DataFrame con las proporciones del target para los valores fuera del rango de confianza.
    """


    result = []  # Lista para almacenar los resultados finales
    
    for var in list_var_continuous:
        # Calcular la media y desviación estándar de la variable
        mean = df[var].mean()
        std = df[var].std()
        
        # Calcular los límites de confianza
        lower_limit = mean - multiplier * std
        upper_limit = mean + multiplier * std
        
        # Filtrar valores fuera del rango
        outliers = df[(df[var] < lower_limit) | (df[var] > upper_limit)]
        
        # Si hay outliers, calcular las proporciones del target
        if not outliers.empty:
            proportions = outliers[target].value_counts(normalize=True)
            proportions = proportions.to_dict()  # Convertir a diccionario para facilitar su uso
            
            # Almacenar la información en una lista
            result.append({
                'variable': var,
                'sum_outlier_values': outliers.shape[0],
                'porcentaje_sum_null_values': outliers.shape[0] / len(df),
                **proportions  # Añadir las proporciones del target
            })
    
    # Si no se encontró ningún outlier, mostrar mensaje
    if not result:
        print('No existen variables con valores fuera del rango de confianza')
    
    # Convertir el resultado en un DataFrame
    result_df = type(df)(result)
    
    return result_df.sort_values(by='sum_outlier_values', ascending=False)

    # %%

def reemplazar_nulos_por_desconocido(df):
    """
    Reemplaza los valores nulos en el DataFrame por "desconocido".
    
    :param df: DataFrame con los datos a procesar.
    :return: DataFrame con los valores nulos reemplazados por "desconocido".
    """
    return df.fillna("desconocido")


    #%%


    # Función para calcular WOE e IV para variables categóricas
def calculate_woe_iv_cat(df, feature, target):
    """""
    Calcula el Weight of Evidence (WoE) y el Information Value (IV) para una variable categórica.
    
    Args:
        df (DataFrame): DataFrame que contiene los datos.
        feature (str): Nombre de la variable categórica.
        target (str): Nombre de la variable objetivo (debe ser binaria: 0/1).

    Returns:
        DataFrame: Tabla con los valores de WoE e IV para cada categoría y el IV total.
    """""
    # Crear una tabla de contingencia para la variable actual
    grouped = df.groupby(feature)[target].value_counts().unstack(fill_value=0)
    
    # Calcular las proporciones de buenos y malos por cada categoría
    grouped['good_pct'] = grouped[1] / grouped[1].sum()
    grouped['bad_pct'] = grouped[0] / grouped[0].sum()

    # Agregar un pequeño valor (epsilon) para evitar división por 0
    epsilon = 1e-6
    grouped['good_pct'] += epsilon
    grouped['bad_pct'] += epsilon

    # Calcular el WOE
    grouped['WOE'] = np.log(grouped['bad_pct'] / grouped['good_pct'])

    # Calcular el IV para cada categoría
    grouped['IV'] = (grouped['bad_pct'] - grouped['good_pct']) * grouped['WOE']

    # Calcular el IV total
    iv_total = grouped['IV'].sum()

    # Agregar una columna con el nombre de la variable y el IV total
    grouped['Feature'] = feature
    grouped['IV_Total'] = iv_total

    return grouped[['WOE', 'IV', 'Feature', 'IV_Total']]
# %%
