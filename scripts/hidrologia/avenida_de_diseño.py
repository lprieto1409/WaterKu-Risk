import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

def calcular_tiempo_concentracion(longitud_km, pendiente_porcentaje, cn):
    """
    Calcula el tiempo de concentración usando la fórmula SCS
    """
    L_metros = longitud_km * 1000
    S_porcentaje = pendiente_porcentaje
    tc_horas = (4.3611 * (L_metros**0.8) * ((1000/cn - 9)**0.7)) / (1900 * (S_porcentaje**0.5))
    return tc_horas

def leer_parametros_cuenca(parametros_excel_path):
    """
    Lee los parámetros geomorfológicos de la cuenca
    """
    try:
        df_parametros = pd.read_excel(parametros_excel_path, index_col=0)
        area_km2 = df_parametros.loc["Área (Km²)", "Valor"]
        longitud_km = df_parametros.loc["Longitud del cauce principal (Km)", "Valor"]
        pendiente_porcentaje = df_parametros.loc["Pendiente media de la cuenca (%)", "Valor"]
        
        print(f"Parámetros leídos:")
        print(f"  Área: {area_km2:.3f} km²")
        print(f"  Longitud cauce: {longitud_km:.3f} km")
        print(f"  Pendiente: {pendiente_porcentaje:.3f} %")
        
        return area_km2, longitud_km, pendiente_porcentaje
    except Exception as e:
        print(f"Error al leer parámetros de la cuenca: {e}")
        return None, None, None

def leer_datos_precipitacion(file_path):
    """Lee los datos de precipitación máxima por períodos de retorno"""
    df = pd.read_excel(file_path)
    return df

def calcular_s_retencion(CN):
    """Calcula la retención potencial S"""
    return (25400 / CN) - 254

def calcular_ia(S):
    """Calcula las abstracciones iniciales Ia"""
    return 0.2 * S

def calcular_escorrentia_efectiva(P, S, Ia):
    """Calcula la escorrentía efectiva usando el método SCS"""
    if P <= Ia:
        return 0
    else:
        return ((P - Ia) ** 2) / (P - Ia + S)

def calcular_tp(tc, dt_horas=1.0):
    """Calcula el tiempo al pico usando tc"""
    return dt_horas / 2 + 0.6 * tc

def calcular_qp(Pe, A, tp):
    """Calcula el caudal pico usando la fórmula SCS"""
    return (0.208 * A * Pe) / tp

def hidrograma_unitario_adimensional_scs(t_tp_ratio):
    """
    Hidrograma unitario adimensional SCS basado en los valores tabulados oficiales
    """
    t_tp_values = [
        0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0
    ]
    
    q_qp_values = [
        0, 0.015, 0.075, 0.16, 0.28, 0.43, 0.6, 0.77, 0.89, 0.97, 1, 0.98, 0.92, 0.84,
        0.75, 0.65, 0.57, 0.43, 0.32, 0.24, 0.18, 0.13, 0.098, 0.075, 0.036, 0.018, 0.009, 0.004
    ]
    
    tabla_scs = dict(zip(t_tp_values, q_qp_values))
    
    if t_tp_ratio in tabla_scs:
        return tabla_scs[t_tp_ratio]
    
    if t_tp_ratio < 0 or t_tp_ratio > max(t_tp_values):
        return 0.0
    
    for i in range(len(t_tp_values) - 1):
        if t_tp_values[i] <= t_tp_ratio <= t_tp_values[i + 1]:
            t1, t2 = t_tp_values[i], t_tp_values[i + 1]
            q1, q2 = q_qp_values[i], q_qp_values[i + 1]
            q_interpolado = q1 + (q2 - q1) * (t_tp_ratio - t1) / (t2 - t1)
            return q_interpolado
    
    return 0.0

def calcular_hidrograma_unitario(Pe, A, tp, duracion_horas=72):
    """Calcula el hidrograma unitario para una precipitación efectiva dada"""
    t_tp_values = [
        0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0
    ]
    
    qp = calcular_qp(Pe, A, tp)
    
    tiempos_scs = []
    caudales_scs = []
    
    for t_tp in t_tp_values:
        t = t_tp * tp
        q_qp_ratio = hidrograma_unitario_adimensional_scs(t_tp)
        caudal = qp * q_qp_ratio
        tiempos_scs.append(t)
        caudales_scs.append(caudal)
    
    return np.array(tiempos_scs), np.array(caudales_scs)

def calcular_caudales_diseño(parametros_excel_path, df_precipitacion, cn, output_dir, excel_filename):
    """Calcula los caudales de diseño para diferentes períodos de retorno"""
    area_km2, longitud_km, pendiente_porcentaje = leer_parametros_cuenca(parametros_excel_path)
    if area_km2 is None:
        raise ValueError("No se pudieron leer los parámetros de la cuenca")
    
    t_conc = calcular_tiempo_concentracion(longitud_km, pendiente_porcentaje, cn)
    print(f"Tiempo de concentración calculado: {t_conc:.3f} horas")
    
    S = calcular_s_retencion(cn)
    Ia = calcular_ia(S)
    tp = calcular_tp(t_conc)
    
    print(f"Parámetros de la cuenca:")
    print(f"  Área: {area_km2} km²")
    print(f"  CN: {cn}")
    print(f"  tc: {t_conc:.3f} horas")
    print(f"  S: {S:.2f} mm")
    print(f"  Ia: {Ia:.2f} mm")
    print(f"  tp: {tp:.2f} horas")
    
    resultados_resumen = []
    hidrogramas_por_tr = {}
    
    for index, row in df_precipitacion.iterrows():
        Tr = int(row['Tr_años'])
        P = row['Precipitacion_Maxima_mm']
        
        Pe = calcular_escorrentia_efectiva(P, S, Ia)
        
        if Pe > 0:
            tiempos, caudales = calcular_hidrograma_unitario(Pe, area_km2, tp)
            
            qp_calculado = np.max(caudales)
            tiempo_pico = tiempos[np.argmax(caudales)]
            
            print(f"Tr = {Tr} años: P = {P:.2f} mm, Pe = {Pe:.2f} mm, Qp = {qp_calculado:.2f} m³/s")
            
            resultado_resumen = {
                'Tr_años': Tr,
                'Precipitacion_mm': P,
                'Escorrentia_efectiva_mm': Pe,
                'Tp_horas': tp,
                'Qp_m3s': qp_calculado
            }
            resultados_resumen.append(resultado_resumen)
            
            hidrogramas_por_tr[Tr] = {
                'tiempos': tiempos,
                'caudales': caudales
            }
    
    if hidrogramas_por_tr:
        primer_tr = list(hidrogramas_por_tr.keys())[0]
        tiempos_comunes = hidrogramas_por_tr[primer_tr]['tiempos']
        
        df_hidrogramas = pd.DataFrame()
        df_hidrogramas['t(hr)'] = np.round(tiempos_comunes, 2)
        
        for tr in sorted(hidrogramas_por_tr.keys()):
            col_name = f'Q(m3/s)'
            caudales_redondeados = np.round(hidrogramas_por_tr[tr]['caudales'], 4)
            df_hidrogramas[col_name] = caudales_redondeados
            
            if tr == sorted(hidrogramas_por_tr.keys())[0]:
                tr_row = [''] + [f'Tr={tr_val}años' for tr_val in sorted(hidrogramas_por_tr.keys())]
                
        df_resumen = pd.DataFrame(resultados_resumen)
        
        df_resumen['Qp_m3s'] = np.round(df_resumen['Qp_m3s'], 4)
        df_resumen['Tp_horas'] = np.round(df_resumen['Tp_horas'], 2)
        if 'Precipitacion_mm' in df_resumen.columns:
            df_resumen['Precipitacion_mm'] = np.round(df_resumen['Precipitacion_mm'], 2)
        if 'Escorrentia_efectiva_mm' in df_resumen.columns:
            df_resumen['Escorrentia_efectiva_mm'] = np.round(df_resumen['Escorrentia_efectiva_mm'], 2)
        
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, excel_filename)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_temp = pd.DataFrame()
            df_temp['t(hr)'] = df_hidrogramas['t(hr)']
            
            for i, tr in enumerate(sorted(hidrogramas_por_tr.keys())):
                col_name = f'Q(m3/s)_Tr{tr}'
                df_temp[col_name] = np.round(hidrogramas_por_tr[tr]['caudales'], 4)
            
            df_temp.to_excel(writer, sheet_name='Hidrogramas', index=False)
            
            worksheet = writer.sheets['Hidrogramas']
            
            worksheet.insert_rows(1)
            worksheet['A1'] = ''
            
            for i, tr in enumerate(sorted(hidrogramas_por_tr.keys())):
                col_letter = chr(66 + i)
                worksheet[f'{col_letter}1'] = f'Tr={tr}años'
            
            worksheet['A2'] = 't(hr)'
            for i, tr in enumerate(sorted(hidrogramas_por_tr.keys())):
                col_letter = chr(66 + i)
                worksheet[f'{col_letter}2'] = 'Q(m3/s)'
            
            df_resumen.to_excel(writer, sheet_name='Resumen', index=False)
        
        print(f"Excel guardado: {output_file}")
        print(f"  - Hoja 'Hidrogramas': {len(df_hidrogramas)} filas de tiempo con formato mejorado")
        print(f"  - Hoja 'Resumen': {len(df_resumen)} períodos de retorno")
        
        return df_resumen, tiempos_comunes, hidrogramas_por_tr
    else:
        print("No se generaron hidrogramas (Pe = 0 para todos los Tr)")
        return pd.DataFrame(), np.array([]), {}

def graficar_avenida_diseno(df_resultados, tiempos, hidrogramas_por_tr, output_dir, grafico_filename):
    """Genera gráfico de avenidas de diseño para diferentes períodos de retorno"""
    plt.figure(figsize=(14, 8))
    
    colores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    
    for i, (tr, datos) in enumerate(sorted(hidrogramas_por_tr.items())):
        tiempos_tr = datos['tiempos']
        caudales_tr = datos['caudales']
        
        qp = np.max(caudales_tr)
        
        plt.plot(tiempos_tr, caudales_tr, 
                linewidth=2, 
                color=colores[i % len(colores)],
                label=f'Tr = {tr} años (Qp = {qp:.1f} m³/s)')
    
    plt.title('Avenidas de Diseño - Hidrograma Unitario SCS', fontsize=14, fontweight='bold')
    plt.xlabel('Tiempo (horas)', fontsize=12)
    plt.ylabel('Caudal (m³/s)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 24)
    plt.ylim(0, None)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, grafico_filename), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Gráfico guardado en: {output_dir}/{grafico_filename}")

def main():
    """Función principal que ejecuta el análisis completo"""
    with open('config/config.yaml', 'r') as file:
        main_config = yaml.safe_load(file)
    
    with open(main_config['basin_config'], 'r') as file:
        basin_config = yaml.safe_load(file)
    
    parametros_excel_path = basin_config['parametros_excel']
    input_precip_path = os.path.join(
        basin_config['analisis_precipitacion_dir'], 
        basin_config['precipitacion_maxima_resultados_excel']
    )
    cn = basin_config['cn_default']
    output_dir = basin_config['avenida_diseno_output_dir']
    
    print("=== ANÁLISIS DE HIDROGRAMA UNITARIO SCS ===")
    print(f"Archivo de parámetros: {parametros_excel_path}")
    print(f"Archivo de precipitación: {input_precip_path}")
    print(f"Directorio de salida: {output_dir}")
    
    df_precipitacion = leer_datos_precipitacion(input_precip_path)
    print(f"\nDatos de precipitación cargados: {len(df_precipitacion)} períodos de retorno")
    
    print("\n=== CÁLCULO DE CAUDALES DE DISEÑO ===")
    df_resultados, tiempos, hidrogramas_por_tr = calcular_caudales_diseño(
        parametros_excel_path, df_precipitacion, cn, output_dir, basin_config['avenida_diseno_excel']
    )
    
    if len(hidrogramas_por_tr) > 0:
        print("\n=== GENERACIÓN DE GRÁFICOS ===")
        graficar_avenida_diseno(df_resultados, tiempos, hidrogramas_por_tr, output_dir, basin_config['avenida_diseno_grafico'])
        
        print("\n=== RESUMEN DE RESULTADOS ===")
        print(f"Archivo Excel generado: {output_dir}/{basin_config['avenida_diseno_excel']}")
        print(f"  - Hoja 'Hidrogramas': Formato t(hr) | Q(m3/s) para cada Tr")
        print(f"  - Hoja 'Resumen': Parámetros y caudales pico")
        print(f"Gráfico generado: {output_dir}/{basin_config['avenida_diseno_grafico']}")
        
        print("\nCaudales pico por período de retorno:")
        for _, row in df_resultados.iterrows():
            print(f"  Tr = {int(row['Tr_años']):4d} años: Qp = {row['Qp_m3s']:6.2f} m³/s")
        
        print("\n=== ANÁLISIS COMPLETADO ===")
    else:
        print("\nNo se generaron hidrogramas. Verifique los datos de entrada.")

if __name__ == "__main__":
    main()
