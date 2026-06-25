import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def leer_datos_excel(archivo):
    """Lee los datos de precipitación máxima para diferentes períodos de retorno"""
    df = pd.read_excel(archivo)
    print(f"Datos cargados desde: {archivo}")
    print(f"Períodos de retorno disponibles: {len(df)} valores")
    return df

def calcular_precipitacion_dick_peschke(P24h_dict, duraciones):
    """Calcula la precipitación para diferentes duraciones usando la fórmula de Dick Peschke"""
    resultados = {}
    
    for tr, P24h in P24h_dict.items():
        precipitaciones = []
        for D in duraciones:
            Pd = P24h * (D / 1440)**0.25
            precipitaciones.append(Pd)
        resultados[f'Tr_{tr}_años'] = precipitaciones
    
    df_precipitaciones = pd.DataFrame(resultados, index=duraciones)
    df_precipitaciones.index.name = 'Duración_min'
    
    return df_precipitaciones

def calcular_intensidades(df_precipitaciones):
    """Calcula las intensidades: I = Pd / D (en horas)"""
    df_intensidades = df_precipitaciones.copy()
    
    for col in df_intensidades.columns:
        df_intensidades[col] = df_intensidades[col] / (df_intensidades.index / 60)
    
    return df_intensidades

def obtener_ecuacion_intensidad(df_intensidades, tr_values):
    """Obtiene la ecuación IDF mediante regresión múltiple en escala logarítmica"""
    print("\nRealizando análisis de regresión múltiple...")
    
    X_data = []
    y_data = []
    
    duraciones = df_intensidades.index.values
    
    for i, tr in enumerate(tr_values):
        col_name = f'Tr_{tr}_años'
        if col_name in df_intensidades.columns:
            intensidades = df_intensidades[col_name].values
            
            for j, duracion in enumerate(duraciones):
                X_data.append([np.log(tr), np.log(duracion)])
                y_data.append(np.log(intensidades[j]))
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    reg = LinearRegression().fit(X, y)
    
    m = reg.coef_[0]
    n = -reg.coef_[1]
    K = np.exp(reg.intercept_)
    
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    
    print(f"Ecuación obtenida: I = {K:.3f} * T^{m:.3f} / D^{n:.3f}")
    print(f"R² = {r2:.4f}")
    
    return {
        'K': K,
        'm': m,
        'n': n,
        'R2': r2,
        'ecuacion': f'I = {K:.3f} * T^{m:.3f} / D^{n:.3f}'
    }

def generar_graficos_idf(df_intensidades, tr_values, archivo_imagen):
    """Genera gráficos de las curvas IDF"""
    plt.figure(figsize=(12, 8))
    
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    duraciones_horas = df_intensidades.index / 60
    
    for i, tr in enumerate(tr_values):
        col_name = f'Tr_{tr}_años'
        if col_name in df_intensidades.columns:
            intensidades = df_intensidades[col_name].values
            plt.plot(duraciones_horas, intensidades, 
                    marker='o', linewidth=2, markersize=4,
                    color=colores[i % len(colores)], 
                    label=f'Tr = {tr} años')
    
    plt.xlabel('Duración (horas)', fontsize=12, fontweight='bold')
    plt.ylabel('Intensidad (mm/h)', fontsize=12, fontweight='bold')
    plt.title('Curvas IDF - Método Dick Peschke', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 25)
    plt.ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(archivo_imagen, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Gráfico guardado: {archivo_imagen}")

def guardar_resultados(df_intensidades, ecuacion, tr_values, archivo_excel, archivo_txt):
    """Guarda los resultados en Excel y archivo de texto"""
    with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
        df_intensidades.to_excel(writer, sheet_name='Intensidades_IDF')
    
    with open(archivo_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ECUACIÓN IDF - MÉTODO DICK PESCHKE\n")
        f.write("=" * 60 + "\n\n")
        f.write("Ecuación de la Intensidad: I = K * T^m / D^n\n\n")
        f.write("Parámetros:\n")
        f.write("-" * 30 + "\n")
        f.write(f"K = {ecuacion['K']:.6f}\n")
        f.write(f"m = {ecuacion['m']:.6f}\n")
        f.write(f"n = {ecuacion['n']:.6f}\n")
        f.write(f"R² = {ecuacion['R2']:.6f}\n\n")
        f.write("Donde:\n")
        f.write("- I: Intensidad de precipitación (mm/h)\n")
        f.write("- T: Período de retorno (años)\n")
        f.write("- D: Duración (minutos)\n")
        f.write("- K, m, n: Parámetros de la regresión\n\n")
        f.write(f"Ecuación final:\n")
        f.write(f"{ecuacion['ecuacion']}\n")
    
    print(f"Resultados guardados:")
    print(f"   - {archivo_excel}")
    print(f"   - {archivo_txt}")
    
    return archivo_excel, archivo_txt

# Función para generar hietograma usando el método del bloque alterno
def generar_hietograma_bloque_alterno(ecuacion, tr_especifico=100, duracion_maxima=24):
    """
    Genera un hietograma usando el método del bloque alterno
    
    Args:
        ecuacion: Diccionario con parámetros K, m, n de la ecuación IDF
        tr_especifico: Período de retorno específico (años)
        duracion_maxima: Duración máxima en horas (default: 24)
    
    Returns:
        DataFrame con el hietograma ordenado
    """
    print(f"\nGenerando hietograma para Tr = {tr_especifico} años...")
    
    # Duraciones en minutos (intervalos de 1 hora)
    duraciones_min = np.arange(60, (duracion_maxima + 1) * 60, 60)
    
    # Calcular intensidades usando la ecuación IDF
    intensidades = []
    for D in duraciones_min:
        I = ecuacion['K'] * (tr_especifico ** ecuacion['m']) / (D ** ecuacion['n'])
        intensidades.append(I)
    
    # Calcular precipitaciones totales acumuladas
    precipitaciones_totales = []
    for i, D in enumerate(duraciones_min):
        P_total = intensidades[i] * (D / 60)  # Intensidad * duración en horas
        precipitaciones_totales.append(P_total)
    
    # Calcular precipitaciones incrementales
    precipitaciones_incrementales = []
    for i in range(len(precipitaciones_totales)):
        if i == 0:
            P_incremental = precipitaciones_totales[i]
        else:
            P_incremental = precipitaciones_totales[i] - precipitaciones_totales[i-1]
        precipitaciones_incrementales.append(P_incremental)
    
    # Crear DataFrame inicial
    df_inicial = pd.DataFrame({
        'Duracion_min': duraciones_min,
        'Intensidad_mm_h': intensidades,
        'Precipitacion_Total_mm': precipitaciones_totales,
        'Precipitacion_Incremental_mm': precipitaciones_incrementales
    })
    
    # Aplicar método del bloque alterno
    df_hietograma = aplicar_bloque_alterno(df_inicial, duracion_maxima)
    
    return df_hietograma

def aplicar_bloque_alterno(df_inicial, duracion_maxima):
    """
    Aplica el método del bloque alterno para reordenar las precipitaciones
    
    Args:
        df_inicial: DataFrame con precipitaciones incrementales
        duracion_maxima: Duración máxima en horas
    
    Returns:
        DataFrame con hietograma ordenado por bloque alterno
    """
    print("Aplicando método del bloque alterno...")
    
    # Obtener precipitaciones incrementales ordenadas de mayor a menor
    precipitaciones = df_inicial['Precipitacion_Incremental_mm'].values
    indices_ordenados = np.argsort(precipitaciones)[::-1]  # Ordenar descendente
    
    # Crear secuencia de bloque alterno
    n_intervalos = len(precipitaciones)
    secuencia_bloque_alterno = np.zeros(n_intervalos)
    
    # Colocar el valor máximo en el centro
    centro = n_intervalos // 2
    secuencia_bloque_alterno[centro] = precipitaciones[indices_ordenados[0]]
    
    # Alternar colocación a izquierda y derecha
    izquierda = centro - 1
    derecha = centro + 1
    
    for i in range(1, n_intervalos):
        if i % 2 == 1:  # Números impares van a la izquierda
            if izquierda >= 0:
                secuencia_bloque_alterno[izquierda] = precipitaciones[indices_ordenados[i]]
                izquierda -= 1
            else:
                secuencia_bloque_alterno[derecha] = precipitaciones[indices_ordenados[i]]
                derecha += 1
        else:  # Números pares van a la derecha
            if derecha < n_intervalos:
                secuencia_bloque_alterno[derecha] = precipitaciones[indices_ordenados[i]]
                derecha += 1
            else:
                secuencia_bloque_alterno[izquierda] = precipitaciones[indices_ordenados[i]]
                izquierda -= 1
    
    # Crear DataFrame final con intervalos
    intervalos = []
    for i in range(n_intervalos):
        inicio = i * 60
        fin = (i + 1) * 60
        intervalos.append(f"{inicio}-{fin}")
    
    df_hietograma = pd.DataFrame({
        'Duracion_min': df_inicial['Duracion_min'].values,
        'Intensidad_mm_h': df_inicial['Intensidad_mm_h'].values,
        'Precipitacion_Total_mm': df_inicial['Precipitacion_Total_mm'].values,
        'Precipitacion_Incremental_mm': df_inicial['Precipitacion_Incremental_mm'].values,
        'Intervalo': intervalos,
        'Precipitacion_mm': secuencia_bloque_alterno
    })
    
    print(f"Hietograma generado con {n_intervalos} intervalos")
    print(f"Precipitación total: {np.sum(secuencia_bloque_alterno):.2f} mm")
    
    return df_hietograma

def generar_grafico_hietograma(df_hietograma, tr_especifico, archivo_imagen):
    """Genera gráfico del hietograma"""
    plt.figure(figsize=(14, 8))
    
    intervalos = range(len(df_hietograma))
    precipitaciones = df_hietograma['Precipitacion_mm'].values
    
    plt.bar(intervalos, precipitaciones, width=0.8, color='steelblue', alpha=0.7, edgecolor='black')
    
    plt.xlabel('Intervalo de tiempo (horas)', fontsize=12, fontweight='bold')
    plt.ylabel('Precipitación (mm)', fontsize=12, fontweight='bold')
    plt.title(f'Hietograma - Método del Bloque Alterno (Tr = {tr_especifico} años)', 
              fontsize=14, fontweight='bold')
    
    etiquetas_x = [f"{i}-{i+1}" for i in range(len(df_hietograma))]
    plt.xticks(intervalos, etiquetas_x, rotation=45, ha='right')
    
    for i, v in enumerate(precipitaciones):
        if v > 0.1:
            plt.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(archivo_imagen, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Gráfico del hietograma guardado: {archivo_imagen}")

def guardar_hietograma(df_hietograma, archivo_excel):
    """Guarda el hietograma en un archivo Excel"""
    with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
        df_hietograma.to_excel(writer, sheet_name='Hietograma', index=False)
    
    print(f"Hietograma guardado: {archivo_excel}")
    
    return archivo_excel

# Función principal
def main(archivo_entrada=None, archivo_intensidades=None, archivo_ecuacion=None, 
         archivo_grafico=None, archivo_hietograma=None, archivo_hietograma_png=None, 
         tr_especifico=100):
    """
    Función principal que ejecuta todo el proceso de generación de curvas IDF
    """
    print("=" * 70)
    print("GENERACIÓN DE CURVAS IDF - MÉTODO DICK PESCHKE")
    print("=" * 70)
    
    # Usar valores por defecto si no se proporcionan
    if archivo_entrada is None:
        archivo_entrada = r"C:\Users\Master\Desktop\GitHub\WaterKu\results\hidrologia\analisis_precipitacion\resultados_precipitacion_max.xlsx"
    
    # Leer los datos de precipitación máxima para diferentes Tr
    df = leer_datos_excel(archivo_entrada)
    
    # Extraer datos de precipitación para diferentes Tr
    P24h_dict = {}
    for _, row in df.iterrows():
        tr = row['Tr_años']
        P24h = row['Precipitacion_Maxima_mm']
        P24h_dict[tr] = P24h
    
    print(f"Períodos de retorno procesados: {list(P24h_dict.keys())}")
    
    # Definir duraciones (en minutos) de 1 a 24 horas
    duraciones = np.array([60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 
                          660, 720, 780, 840, 900, 960, 1020, 1080, 1140, 
                          1200, 1260, 1320, 1380, 1440])
    
    tr_values = np.array(list(P24h_dict.keys()))
    
    print(f"Duraciones analizadas: {len(duraciones)} valores (1h a 24h)")
    print(f"Períodos de retorno: {tr_values}")
    
    # Paso 1: Calcular precipitaciones para diferentes duraciones usando Dick Peschke
    print(f"\nAplicando método Dick Peschke: Pd = P24h * (D/1440)^0.25")
    df_precipitaciones = calcular_precipitacion_dick_peschke(P24h_dict, duraciones)
    
    # Paso 2: Calcular intensidades
    print(f"Calculando intensidades: I = Pd / D (horas)")
    df_intensidades = calcular_intensidades(df_precipitaciones)
    
    # Paso 3: Obtener ecuación IDF mediante regresión múltiple
    ecuacion = obtener_ecuacion_intensidad(df_intensidades, tr_values)
    
    # Crear directorios de salida
    if archivo_intensidades:
        os.makedirs(os.path.dirname(archivo_intensidades), exist_ok=True)
    
    # Paso 4: Generar gráficos
    if archivo_grafico:
        print(f"\nGenerando gráficos de curvas IDF...")
        generar_graficos_idf(df_intensidades, tr_values, archivo_grafico)
    
    # Paso 5: Guardar resultados
    if archivo_intensidades and archivo_ecuacion:
        print(f"\nGuardando resultados...")
        guardar_resultados(df_intensidades, ecuacion, tr_values, archivo_intensidades, archivo_ecuacion)
    
    # Paso 6: Generar hietograma
    print(f"\nGenerando hietograma para Tr = {tr_especifico} años...")
    df_hietograma = generar_hietograma_bloque_alterno(ecuacion, tr_especifico, duracion_maxima=24)
    
    # Paso 7: Generar gráfico del hietograma
    if archivo_hietograma_png:
        print(f"\nGenerando gráfico del hietograma...")
        generar_grafico_hietograma(df_hietograma, tr_especifico, archivo_hietograma_png)
    
    # Paso 8: Guardar hietograma en Excel
    if archivo_hietograma:
        print(f"\nGuardando hietograma...")
        guardar_hietograma(df_hietograma, archivo_hietograma)
    
    # Mostrar resumen de resultados
    print(f"\nPROCESO COMPLETADO")
    print("=" * 70)
    print(f"Ecuación IDF obtenida:")
    print(f"   {ecuacion['ecuacion']}")
    print(f"   R² = {ecuacion['R2']:.4f}")
    print(f"\nHietograma generado:")
    print(f"   Período de retorno: {tr_especifico} años")
    print(f"   Duración total: 24 horas")
    print(f"   Precipitación total: {df_hietograma['Precipitacion_mm'].sum():.2f} mm")
    
    return df_intensidades, ecuacion, df_hietograma

# Ejecutar el script
if __name__ == "__main__":
    # Verificar si se está ejecutando desde Snakemake
    try:
        # Si snakemake existe, usar los archivos de entrada y salida definidos en Snakefile
        archivo_entrada = snakemake.input.precipitacion
        archivo_intensidades = snakemake.output.intensidades
        archivo_ecuacion = snakemake.output.ecuacion
        archivo_grafico = snakemake.output.grafico
        archivo_hietograma = snakemake.output.hietograma
        archivo_hietograma_png = snakemake.output.hietograma_png
        
        # Cargar configuración
        with open(snakemake.input.basin_config) as f:
            basin_config = yaml.safe_load(f)
        
        tr_especifico = basin_config.get('curvas_idf_tr_especifico', 100)
        
        df_intensidades, ecuacion, df_hietograma = main(
            archivo_entrada=archivo_entrada,
            archivo_intensidades=archivo_intensidades,
            archivo_ecuacion=archivo_ecuacion,
            archivo_grafico=archivo_grafico,
            archivo_hietograma=archivo_hietograma,
            archivo_hietograma_png=archivo_hietograma_png,
            tr_especifico=tr_especifico
        )
        
    except NameError:
        # Si no se ejecuta desde Snakemake, usar valores por defecto
        df_intensidades, ecuacion, df_hietograma = main()
