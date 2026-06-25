import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import skew

def leer_datos(file_path):
    """Lee los datos de precipitación desde un archivo Excel"""
    df = pd.read_excel(file_path)
    print(f"Estructura del DataFrame:")
    print(f"Columnas: {list(df.columns)}")
    print(f"Primeras 5 filas:")
    print(df.head())
    return df

def completar_datos_faltantes_y_obtener_maximos(df):
    """Completa los datos faltantes usando regresión y obtiene máximos anuales"""
    print("\n=== COMPLETANDO DATOS FALTANTES Y OBTENIENDO MÁXIMOS ANUALES ===")
    if 'Año' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'Año'})
    
    columnas_meses = [col for col in df.columns if col != 'Año']
    print(f"Columnas de meses detectadas: {columnas_meses}")
    
    for mes in columnas_meses:
        df[mes] = pd.to_numeric(df[mes], errors='coerce')
    
    print(f"Datos limpiados: valores 'nn' y otros no numéricos convertidos a NaN")
    
    print("\nResumen de datos faltantes por mes:")
    for mes in columnas_meses:
        faltantes = df[mes].isna().sum()
        if faltantes > 0:
            print(f"  {mes}: {faltantes} valores faltantes")
    
    for mes in columnas_meses:
        datos_faltantes = df[df[mes].isnull()]
        
        if len(datos_faltantes) > 0:
            print(f"\nCompletando {len(datos_faltantes)} datos faltantes para {mes}")
            
            datos_disponibles = df.dropna(subset=[mes])
            
            if len(datos_disponibles) >= 1:
                X = datos_disponibles['Año'].values.reshape(-1, 1)
                y = datos_disponibles[mes].values
                
                modelo = LinearRegression()
                
                if len(datos_disponibles) == 1:
                    print(f"  Solo 1 dato disponible para {mes}, usando regresión horizontal")
                    X_artificial = np.array([[X[0][0]], [X[0][0] + 1]])
                    y_artificial = np.array([y[0], y[0]])
                    modelo.fit(X_artificial, y_artificial)
                else:
                    modelo.fit(X, y)
                
                años_faltantes = datos_faltantes['Año'].values.reshape(-1, 1)
                predicciones = modelo.predict(años_faltantes)
                
                predicciones = np.maximum(predicciones, 0)
                
                df.loc[df[mes].isnull(), mes] = predicciones
                
                print(f"  Años completados para {mes}: {datos_faltantes['Año'].tolist()}")
                print(f"  Valores predichos: {[f'{pred:.2f}' for pred in predicciones]}")
            else:
                print(f"  No hay datos disponibles para {mes}, usando 0.0")
                df.loc[df[mes].isnull(), mes] = 0.0
    
    print("\n=== OBTENIENDO PRECIPITACIÓN MÁXIMA ANUAL ===")
    
    df['Precipitación'] = df[columnas_meses].max(axis=1)
    
    df['Mes_Maximo'] = df[columnas_meses].idxmax(axis=1)
    
    df_final = df[['Año', 'Precipitación', 'Mes_Maximo']].copy()
    
    print(f"Precipitación máxima anual calculada para {len(df_final)} años")
    print(f"Rango de años: {df_final['Año'].min()} - {df_final['Año'].max()}")
    
    print("\nPrimeros 10 valores de precipitación máxima anual:")
    print(df_final.head(10))
    
    return df_final, df

data = [
    (10, 2.036), (11, 2.088), (12, 2.134), (13, 2.175), (14, 2.213),
    (15, 2.247), (16, 2.279), (17, 2.309), (18, 2.335), (19, 2.361),
    (20, 2.385), (21, 2.408), (22, 2.429), (23, 2.448), (24, 2.467),
    (25, 2.486), (26, 2.502), (27, 2.519), (28, 2.534), (29, 2.549),
    (30, 2.563), (31, 2.577), (32, 2.591), (33, 2.604), (34, 2.616),
    (35, 2.628), (36, 2.639), (37, 2.650), (38, 2.661), (39, 2.671),
    (40, 2.682), (41, 2.692), (42, 2.700), (43, 2.710), (44, 2.719),
    (45, 2.727), (46, 2.736), (47, 2.744), (48, 2.753), (49, 2.760),
    (50, 2.768), (55, 2.804), (60, 2.837), (65, 2.866), (70, 2.893),
    (75, 2.917), (80, 2.940), (85, 2.961), (90, 2.981), (95, 3.000),
    (100, 3.017), (110, 3.049), (120, 3.078), (130, 3.104), (140, 3.129)
]

tabla_Kn = pd.DataFrame(data, columns=["Datos", "K_n"])

def obtener_Kn(num_datos):
    """Obtiene el valor K_n según el número de datos usando interpolación si es necesario"""
    if num_datos in tabla_Kn['Datos'].values:
        return tabla_Kn[tabla_Kn['Datos'] == num_datos]['K_n'].values[0]
    
    lower = tabla_Kn[tabla_Kn['Datos'] <= num_datos].iloc[-1]
    upper = tabla_Kn[tabla_Kn['Datos'] > num_datos].iloc[0]
    K_n = lower['K_n'] + (upper['K_n'] - lower['K_n']) * (num_datos - lower['Datos']) / (upper['Datos'] - lower['Datos'])
    return K_n

def calcular_parametros_estadisticos(df):
    """Calcula parámetros estadísticos de los datos de precipitación"""
    stats = {
        'Número de datos': len(df),
        'Sumatoria': df['Precipitación'].sum(),
        'Valor máximo': df['Precipitación'].max(),
        'Valor mínimo': df['Precipitación'].min(),
        'Media': df['Precipitación'].mean(),
        'Varianza': df['Precipitación'].var(),
        'Desviación estándar': df['Precipitación'].std(),
        'Coeficiente de variación': df['Precipitación'].std() / df['Precipitación'].mean(),
        'Coeficiente de sesgo': skew(df['Precipitación'])
    }
    
    log_precipitacion = np.log10(df['Precipitación'].replace(0, np.nan))
    stats['Media log(P24h)'] = log_precipitacion.mean()
    stats['Varianza log(P24h)'] = log_precipitacion.var()
    stats['Desviación estándar log(P24h)'] = log_precipitacion.std()
    stats['Coeficiente de variación log(P24h)'] = log_precipitacion.std() / log_precipitacion.mean()
    stats['Coeficiente de sesgo log(P24h)'] = skew(log_precipitacion.dropna())
    
    return stats

def detectar_outliers(df):
    """Detecta outliers usando la fórmula según asimetría del log(P24h)"""
    num_datos = len(df)
    K_n = obtener_Kn(num_datos)
    
    log_precipitacion = np.log10(df['Precipitación'].replace(0, np.nan))
    media_log = log_precipitacion.mean()
    desviacion_log = log_precipitacion.std()
    coef_sesgo_log = skew(log_precipitacion.dropna())
    
    print(f"Número de datos: {num_datos}")
    print(f"Valor K_n: {K_n:.4f}")
    print(f"Coeficiente de sesgo log(P24h): {coef_sesgo_log:.4f}")
    print(f"Media log(P24h): {media_log:.4f}")
    print(f"Desviación estándar log(P24h): {desviacion_log:.4f}")
    
    df['Outlier'] = False
    outliers_detectados = []
    umbral_info = {}
    
    if coef_sesgo_log > 0.4:
        print("Asimetría > +0.4: Detectando outliers altos primero")
        print(f"Usando K_n = {K_n:.4f} para el cálculo")
        umbral_alto_log = media_log + K_n * desviacion_log
        umbral_alto = 10**umbral_alto_log
        print(f"Cálculo: {media_log:.4f} + {K_n:.4f} * {desviacion_log:.4f} = {umbral_alto_log:.4f}")
        print(f"Umbral alto (log): {umbral_alto_log:.4f} -> Umbral alto: {umbral_alto:.2f} mm")
        
        umbral_info['umbral_alto'] = umbral_alto
        
        outliers_altos = df['Precipitación'] > umbral_alto
        df.loc[outliers_altos, 'Outlier'] = True
        outliers_detectados.extend(df[outliers_altos]['Año'].tolist())
        
    elif coef_sesgo_log < -0.4:
        print("Asimetría < -0.4: Detectando outliers bajos primero")
        print(f"Usando K_n = {K_n:.4f} para el cálculo")
        umbral_bajo_log = media_log - K_n * desviacion_log
        umbral_bajo = 10**umbral_bajo_log
        print(f"Cálculo: {media_log:.4f} - {K_n:.4f} * {desviacion_log:.4f} = {umbral_bajo_log:.4f}")
        print(f"Umbral bajo (log): {umbral_bajo_log:.4f} -> Umbral bajo: {umbral_bajo:.2f} mm")
        
        umbral_info['umbral_bajo'] = umbral_bajo
        
        outliers_bajos = df['Precipitación'] < umbral_bajo
        df.loc[outliers_bajos, 'Outlier'] = True
        outliers_detectados.extend(df[outliers_bajos]['Año'].tolist())
        
    else:
        print("Asimetría entre -0.4 y +0.4: Detectando outliers altos y bajos")
        print(f"Usando K_n = {K_n:.4f} para el cálculo")
        umbral_alto_log = media_log + K_n * desviacion_log
        umbral_bajo_log = media_log - K_n * desviacion_log
        umbral_alto = 10**umbral_alto_log
        umbral_bajo = 10**umbral_bajo_log
        
        print(f"Cálculo umbral alto: {media_log:.4f} + {K_n:.4f} * {desviacion_log:.4f} = {umbral_alto_log:.4f}")
        print(f"Cálculo umbral bajo: {media_log:.4f} - {K_n:.4f} * {desviacion_log:.4f} = {umbral_bajo_log:.4f}")
        print(f"Umbral alto (log): {umbral_alto_log:.4f} -> Umbral alto: {umbral_alto:.2f} mm")
        print(f"Umbral bajo (log): {umbral_bajo_log:.4f} -> Umbral bajo: {umbral_bajo:.2f} mm")
        
        umbral_info['umbral_alto'] = umbral_alto
        umbral_info['umbral_bajo'] = umbral_bajo
        
        outliers_altos = df['Precipitación'] > umbral_alto
        outliers_bajos = df['Precipitación'] < umbral_bajo
        
        df.loc[outliers_altos | outliers_bajos, 'Outlier'] = True
        outliers_detectados.extend(df[outliers_altos | outliers_bajos]['Año'].tolist())
    
    if outliers_detectados:
        print(f"Outliers detectados en los años: {outliers_detectados}")
        print(f"Total de outliers: {len(outliers_detectados)}")
        for año in outliers_detectados:
            valor = df[df['Año'] == año]['Precipitación'].values[0]
            print(f"  Año {año}: {valor:.2f} mm")
    else:
        print("No se detectaron outliers")
    
    return df, umbral_info

def eliminar_outliers_y_repetir(df_maximos, df_completo):
    """Repite el análisis hasta que no haya outliers"""
    iteracion = 1
    print("\n=== PROCESO DE ELIMINACIÓN DE OUTLIERS ===")
    umbral_final = None
    
    while True:
        print(f"\nIteración {iteracion}:")
        print(f"Número de datos: {len(df_maximos)}")
        
        df_temp, umbral_info = detectar_outliers(df_maximos.copy())
        
        umbral_final = umbral_info
        
        outliers_encontrados = df_temp['Outlier'].sum()
        if outliers_encontrados == 0:
            print("No se encontraron más outliers. Proceso terminado.")
            break
        
        años_outliers = df_temp[df_temp['Outlier']]['Año'].tolist()
        print(f"Años con outliers: {años_outliers}")
        
        for año in años_outliers:
            mes_maximo = df_maximos[df_maximos['Año'] == año]['Mes_Maximo'].iloc[0]
            valor_original = df_completo.loc[df_completo['Año'] == año, mes_maximo].iloc[0]
            
            print(f"  Año {año}: Colocando 0 en {mes_maximo} (valor original: {valor_original:.2f})")
            
            df_completo.loc[df_completo['Año'] == año, mes_maximo] = 0.0
        
        print("Recalculando máximos anuales...")
        columnas_meses = [col for col in df_completo.columns if col not in ['Año', 'Precipitación', 'Mes_Maximo']]
        
        df_completo['Precipitación'] = df_completo[columnas_meses].max(axis=1)
        df_completo['Mes_Maximo'] = df_completo[columnas_meses].idxmax(axis=1)
        
        df_maximos = df_completo[['Año', 'Precipitación', 'Mes_Maximo']].copy()
        
        for año in años_outliers:
            nuevo_valor = df_maximos[df_maximos['Año'] == año]['Precipitación'].iloc[0]
            nuevo_mes = df_maximos[df_maximos['Año'] == año]['Mes_Maximo'].iloc[0]
            print(f"  Año {año}: Nuevo máximo = {nuevo_valor:.2f} mm (mes: {nuevo_mes})")
        
        print(f"Outliers procesados: {outliers_encontrados}")
        print(f"Datos restantes: {len(df_maximos)}")
        
        iteracion += 1
        
        if iteracion > 10:
            print("Máximo número de iteraciones alcanzado.")
            break
    
    return df_maximos, umbral_final

def generar_output(df, stats, output_excel, output_grafico, umbral_info=None):
    """Genera el output: archivo Excel y gráfico"""
    df_output = df[['Año', 'Precipitación']].copy()
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        df_output.to_excel(writer, sheet_name='Datos_Procesados', index=False)
    
    plt.figure(figsize=(12, 8))
    plt.plot(df['Año'], df['Precipitación'], 'o-', linewidth=2, markersize=6, color='blue', label='Precipitación Máxima Anual')
    
    if umbral_info:
        if 'umbral_alto' in umbral_info:
            plt.axhline(y=umbral_info['umbral_alto'], color='red', linestyle='--', linewidth=2, 
                       label=f'Umbral Superior: {umbral_info["umbral_alto"]:.1f} mm')
        if 'umbral_bajo' in umbral_info:
            plt.axhline(y=umbral_info['umbral_bajo'], color='orange', linestyle='--', linewidth=2,
                       label=f'Umbral Inferior: {umbral_info["umbral_bajo"]:.1f} mm')
    
    plt.title('Precipitación Máxima 24h Anual', fontsize=16, fontweight='bold')
    plt.xlabel('Año', fontsize=12)
    plt.ylabel('Precipitación (mm)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_grafico, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Archivo Excel generado: {output_excel}")
    print(f"Gráfico generado: {output_grafico}")

def generar_reporte_texto(df, stats, output_reporte, valores_completados):
    """Genera reporte de texto"""
    with open(output_reporte, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("REPORTE DE ANÁLISIS DE PRECIPITACIÓN MÁXIMA 24H ANUAL\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("INFORMACIÓN GENERAL:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Período analizado: {df['Año'].min()} - {df['Año'].max()}\n")
        f.write(f"Valores completados: {valores_completados}\n")
        f.write(f"Datos finales: {len(df)} registros\n\n")
        
        f.write("PARÁMETROS ESTADÍSTICOS:\n")
        f.write("-" * 30 + "\n")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")

def main(input_file=None, output_excel=None, output_grafico=None, output_reporte=None):
    """Función principal"""
    if input_file is None:
        input_file = r'C:\Users\Master\Desktop\GitHub\WaterKu\data\precipitacion\precipitacion_max_24h.xlsx'
    if output_excel is None:
        output_excel = r'C:\Users\Master\Desktop\GitHub\WaterKu\results\hidrologia\analisis_precipitacion\precipitacion_maxima_anual.xlsx'
    if output_grafico is None:
        output_grafico = r'C:\Users\Master\Desktop\GitHub\WaterKu\results\hidrologia\analisis_precipitacion\precipitacion_maxima_anual_grafico.png'
    if output_reporte is None:
        output_reporte = r'C:\Users\Master\Desktop\GitHub\WaterKu\results\hidrologia\analisis_precipitacion\precipitacion_maxima_anual_reporte.txt'
    
    import os
    output_dir = os.path.dirname(output_excel)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directorio de salida: {output_dir}")
    
    print("=== ANÁLISIS DE PRECIPITACIÓN MÁXIMA 24H ANUAL ===")
    print(f"Archivo de entrada: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"ERROR: El archivo de entrada no existe: {input_file}")
        return
    
    df = leer_datos(input_file)
    print(f"Datos originales cargados: {len(df)} registros")
    
    df_original = df.copy()
    df_maximos, df_completo = completar_datos_faltantes_y_obtener_maximos(df)
    
    valores_completados = 0
    for col in df_original.columns:
        if col != 'Año' and col in df_original.columns:
            valores_completados += df_original[col].isna().sum()
    
    if valores_completados > 0:
        print(f"Valores faltantes completados: {valores_completados}")
    
    df_maximos, umbral_final = eliminar_outliers_y_repetir(df_maximos, df_completo)
    
    stats = calcular_parametros_estadisticos(df_maximos)
    
    generar_output(df_maximos, stats, output_excel, output_grafico, umbral_final)
    
    generar_reporte_texto(df_maximos, stats, output_reporte, valores_completados)
    
    print(f"\n=== PROCESO COMPLETADO ===")
    print(f"Archivos generados:")
    print(f"  - {output_excel}")
    print(f"  - {output_grafico}")
    print(f"  - {output_reporte}")

if __name__ == "__main__":
    main()
