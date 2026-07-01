import numpy as np
import pandas as pd
import scipy.stats as stats
import os
import matplotlib.pyplot as plt
import yaml

def leer_datos_excel(archivo):
    """Lee datos de precipitación desde un archivo Excel"""
    df = pd.read_excel(archivo)
    data = df['Precipitación'].sort_values() 
    return data

def calcular_parametros_estadisticos(data):
    """Calcula parámetros estadísticos de los datos"""
    media = np.mean(data)
    varianza = np.var(data)
    des_estandar = np.std(data)
    coef_variacion = des_estandar / media
    sesgo = stats.skew(data)
    curtosis = stats.kurtosis(data)
    
    parametros = {
        'media': media,
        'varianza': varianza,
        'des.estandar': des_estandar,
        'coef.variacion': coef_variacion,
        'coef.sesgo': sesgo,
        'coef.curtosis': curtosis
    }
    return parametros

def ajustar_distribuciones(data):
    """Ajusta diferentes distribuciones estadísticas y evalúa su bondad de ajuste"""
    print("Ajustando distribuciones...")
    
    dist_normal = stats.norm.fit(data)
    dist_lognorm_2 = stats.lognorm.fit(data, floc=0)
    dist_lognorm_3 = stats.lognorm.fit(data)
    dist_gumbel = stats.gumbel_r.fit(data)
    dist_loggumbel = stats.gumbel_r.fit(np.log(data))
    dist_pearson3 = stats.genextreme.fit(data)
    dist_logpearson3 = stats.genextreme.fit(np.log(data))
    
    print("Evaluando bondad de ajuste...")
    ks_test = {
        'normal': stats.kstest(data, 'norm', args=dist_normal),
        'lognorm_2': stats.kstest(data, 'lognorm', args=dist_lognorm_2),
        'lognorm_3': stats.kstest(data, 'lognorm', args=dist_lognorm_3),
        'gumbel': stats.kstest(data, 'gumbel_r', args=dist_gumbel),
        'loggumbel': stats.kstest(np.log(data), 'gumbel_r', args=dist_loggumbel),
        'pearson3': stats.kstest(data, 'genextreme', args=dist_pearson3),
        'logpearson3': stats.kstest(np.log(data), 'genextreme', args=dist_logpearson3),
    }
    
    print("\nResultados de las pruebas K-S:")
    for dist, result in ks_test.items():
        print(f"  {dist.ljust(12)}: p-valor = {result[1]:.6f}")
    
    return ks_test, dist_normal, dist_lognorm_2, dist_lognorm_3, dist_gumbel, dist_loggumbel, dist_pearson3, dist_logpearson3

def calcular_precipitacion_max(tr, dist, tipo_dist, data):
    """Calcula la precipitación máxima para un período de retorno dado usando probabilidad de Weibull"""
    weibull_prob = 1 - (1 / tr)
    
    try:
        if tipo_dist == 'normal':
            p_max = stats.norm.ppf(weibull_prob, loc=dist[0], scale=dist[1])
        elif tipo_dist in ['lognorm_2', 'lognorm_3']:
            p_max = stats.lognorm.ppf(weibull_prob, dist[0], dist[1], dist[2])
        elif tipo_dist == 'gumbel':
            p_max = stats.gumbel_r.ppf(weibull_prob, loc=dist[0], scale=dist[1])
        elif tipo_dist == 'loggumbel':
            log_p_max = stats.gumbel_r.ppf(weibull_prob, loc=dist[0], scale=dist[1])
            p_max = np.exp(log_p_max)
        elif tipo_dist in ['pearson3', 'logpearson3']:
            if tipo_dist == 'pearson3':
                p_max = stats.genextreme.ppf(weibull_prob, dist[0], dist[1], dist[2])
            else:
                log_p_max = stats.genextreme.ppf(weibull_prob, dist[0], dist[1], dist[2])
                p_max = np.exp(log_p_max)
        else:
            print(f"Advertencia: Tipo de distribución '{tipo_dist}' no reconocido, usando normal como respaldo")
            media = np.mean(data)
            std = np.std(data)
            p_max = stats.norm.ppf(weibull_prob, loc=media, scale=std)
            
        return p_max
        
    except Exception as e:
        print(f"Error calculando precipitación máxima para Tr={tr}: {e}")
        media = np.mean(data)
        std = np.std(data)
        return stats.norm.ppf(weibull_prob, loc=media, scale=std)

def elegir_mejor_distribucion(ks_test):
    """Selecciona la distribución con mejor ajuste según la prueba K-S"""
    mejor_ajuste = max(ks_test, key=lambda k: ks_test[k][1])
    print(f"Mejor distribución encontrada: {mejor_ajuste}")
    print(f"P-valor del test K-S: {ks_test[mejor_ajuste][1]:.6f}")
    
    return mejor_ajuste, ks_test[mejor_ajuste]

def guardar_resultados(archivo_excel, archivo_txt, parametros, ks_test, mejor_distribucion, p_max):
    """Guarda los resultados en archivos Excel y de texto"""
    precipitacion_data = []
    for tr, precip in p_max.items():
        precipitacion_data.append({
            'Tr_años': tr,
            'Precipitacion_Maxima_mm': precip
        })
    
    df_precipitacion = pd.DataFrame(precipitacion_data)
    
    with pd.ExcelWriter(archivo_excel, engine='openpyxl') as writer:
        df_precipitacion.to_excel(writer, sheet_name='Precipitacion_Maxima', index=False)
    with open(archivo_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ANÁLISIS DE PRECIPITACIÓN MÁXIMA\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PARÁMETROS ESTADÍSTICOS:\n")
        f.write("-" * 30 + "\n")
        for key, value in parametros.items():
            f.write(f"{key.ljust(20)}: {value:.6f}\n")
        
        f.write("\nPRUEBAS DE KOLMOGOROV-SMIRNOV:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Distribución'.ljust(15)} {'Estadístico'.ljust(12)} {'P-valor'.ljust(12)}\n")
        f.write("-" * 40 + "\n")
        for dist, result in ks_test.items():
            f.write(f"{dist.ljust(15)} {result[0]:11.6f} {result[1]:11.6f}\n")
        
        f.write(f"\nMEJOR DISTRIBUCIÓN: {mejor_distribucion.upper()}\n")
        f.write(f"P-valor: {ks_test[mejor_distribucion][1]:.6f}\n")
        
        f.write(f"\nPRECIPITACIÓN MÁXIMA POR PERÍODO DE RETORNO:\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'Tr (años)'.ljust(12)} {'Precipitación (mm)'.ljust(20)}\n")
        f.write("-" * 45 + "\n")
        for tr, precip in p_max.items():
            f.write(f"{tr:8d} {precip:15.2f}\n")
        
        f.write(f"\nNOTA: Los valores están calculados usando la distribución {mejor_distribucion}\n")
        f.write(f"que mostró el mejor ajuste según la prueba de Kolmogorov-Smirnov.\n")

def main(archivo_datos=None, archivo_resultados_excel=None, archivo_resultados_txt=None):
    """Función principal que ejecuta el análisis completo de precipitación máxima"""
    if archivo_datos is None:
        archivo_datos = r"C:\Users\Master\Desktop\GitHub\WaterKu\results\hidrologia\analisis_precipitacion\precipitacion_maxima_anual.xlsx"
    if archivo_resultados_excel is None:
        archivo_resultados_excel = r"C:\Users\Master\Desktop\GitHub\WaterKu\results\hidrologia\analisis_precipitacion\resultados_precipitacion_max.xlsx"
    if archivo_resultados_txt is None:
        archivo_resultados_txt = r"C:\Users\Master\Desktop\GitHub\WaterKu\results\hidrologia\analisis_precipitacion\resultados_precipitacion_max.txt"
    
    os.makedirs(os.path.dirname(archivo_resultados_excel), exist_ok=True)
    
    data = leer_datos_excel(archivo_datos)
    print(f"Datos cargados: {len(data)} valores")
    print(f"Rango de precipitación: {data.min():.2f} - {data.max():.2f} mm")
    
    parametros = calcular_parametros_estadisticos(data)
    print(f"Parámetros estadísticos calculados")
    
    ks_test, dist_normal, dist_lognorm_2, dist_lognorm_3, dist_gumbel, dist_loggumbel, dist_pearson3, dist_logpearson3 = ajustar_distribuciones(data)
    print(f"Distribuciones ajustadas")
    
    mejor_distribucion, dist_ajustada = elegir_mejor_distribucion(ks_test)
    
    distribuciones_params = {
        'normal': dist_normal,
        'lognorm_2': dist_lognorm_2,
        'lognorm_3': dist_lognorm_3,
        'gumbel': dist_gumbel,
        'loggumbel': dist_loggumbel,
        'pearson3': dist_pearson3,
        'logpearson3': dist_logpearson3
    }
    
    parametros_mejor_dist = distribuciones_params[mejor_distribucion]
    
    p_max = {}
    tr_values = [2, 5, 10, 25, 50, 100, 200, 500, 1000]
    
    print(f"\nCalculando precipitación máxima para diferentes períodos de retorno:")
    for tr in tr_values:
        p_max[tr] = calcular_precipitacion_max(tr, parametros_mejor_dist, mejor_distribucion, data)
        print(f"Tr = {tr:4d} años: {p_max[tr]:7.2f} mm")
    
    guardar_resultados(archivo_resultados_excel, archivo_resultados_txt, parametros, ks_test, mejor_distribucion, p_max)
    print(f"\nResultados guardados en:")
    print(f"  - {archivo_resultados_excel}")
    print(f"  - {archivo_resultados_txt}")

if __name__ == "__main__":
    try:
        archivo_datos = snakemake.input.precipitacion
        archivo_resultados_excel = snakemake.output.excel
        archivo_resultados_txt = snakemake.output.reporte
        
        with open(snakemake.input.basin_config) as f:
            basin_config = yaml.safe_load(f)
        
        main(archivo_datos, archivo_resultados_excel, archivo_resultados_txt)
        
    except NameError:
        resultados_dir = r"C:\Users\Master\Desktop\GitHub\WaterKu\results\hidrologia\analisis_precipitacion"
        os.makedirs(resultados_dir, exist_ok=True)
        
        main()
