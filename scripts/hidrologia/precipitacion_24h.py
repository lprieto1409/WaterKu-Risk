import pandas as pd
import numpy as np
import os
import glob
import yaml

def procesar_archivo_precipitacion(ruta_archivo):
    """
    Procesa un archivo de precipitación SENAHMI y genera un Excel 
    con precipitación máxima de 24h organizada por año y mes
    """
    try:
        carpeta_base = os.path.dirname(ruta_archivo)
        nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]
        archivo_salida = os.path.join(carpeta_base, f"{nombre_archivo}_precip_max_24h.xlsx")
        
        print(f"\n--- Procesando: {os.path.basename(ruta_archivo)} ---")
        
        df = pd.read_csv(ruta_archivo, sep=r'\s+', header=None)
        columnas_esperadas = ['Año', 'Mes', 'Día', 'Precipitación acumulada', 'Temperatura máxima', 'Temperatura mínima']
        df.columns = columnas_esperadas[:len(df.columns)]
        registros_validos = df[df['Precipitación acumulada'] != -99.9]       
        df.replace(-99.9, np.nan, inplace=True)
        
        if 'Año' not in df.columns or 'Mes' not in df.columns or 'Día' not in df.columns:
            raise ValueError("El archivo no tiene las columnas básicas de fecha (Año, Mes, Día)")
        
        df = df[
            (df['Año'] >= 1900) & (df['Año'] <= 2100) &
            (df['Mes'] >= 1) & (df['Mes'] <= 12) &
            (df['Día'] >= 1) & (df['Día'] <= 31)
        ]
        df['Fecha'] = (df['Año'].astype(str) + '-' + 
                      df['Mes'].astype(str).str.zfill(2) + '-' + 
                      df['Día'].astype(str).str.zfill(2))
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df = df.dropna(subset=['Fecha'])
        
        df['Año-Mes'] = df['Fecha'].dt.to_period('M')
        precip_max_24h = df.groupby(['Año', 'Mes'])['Precipitación acumulada'].max().reset_index()
        tabla_pivote = precip_max_24h.pivot(index='Año', columns='Mes', values='Precipitación acumulada')
        
        nombres_meses = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        tabla_pivote = tabla_pivote.rename(columns=nombres_meses)
        
        meses_ordenados = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                          'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        columnas_existentes = [mes for mes in meses_ordenados if mes in tabla_pivote.columns]
        tabla_pivote = tabla_pivote[columnas_existentes]
        
        with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
            tabla_pivote.to_excel(writer, sheet_name='Precipitacion_Max_24h', index_label='Año')
            df_muestra = df.head(100)  
            df_muestra.to_excel(writer, sheet_name='Datos_Originales_Muestra', index=False)
        
        print(f"  Excel generado: {os.path.basename(archivo_salida)}")
        print(f"  Años procesados: {tabla_pivote.index.min()} - {tabla_pivote.index.max()}")
        print(f"  Meses con datos: {len(columnas_existentes)}")
        
        return tabla_pivote, archivo_salida, True
        
    except Exception as e:
        print(f"  Error al procesar {os.path.basename(ruta_archivo)}: {e}")
        return None, None, False

def procesar_carpeta_completa(ruta_carpeta):
    """
    Procesa todos los archivos .txt de una carpeta
    """
    patron_busqueda = os.path.join(ruta_carpeta, "*.txt")
    archivos_txt = glob.glob(patron_busqueda)
    
    if not archivos_txt:
        print(f"No se encontraron archivos .txt en la carpeta: {ruta_carpeta}")
        return
    
    print(f"Carpeta a procesar: {ruta_carpeta}")
    print(f"Archivos encontrados: {len(archivos_txt)}")
    for i, archivo in enumerate(archivos_txt, 1):
        print(f"   {i}. {os.path.basename(archivo)}")
    
    print(f"\n{'='*60}")
    print(f"INICIANDO PROCESAMIENTO DE {len(archivos_txt)} ARCHIVO(S)")
    print(f"{'='*60}")
    
    exitosos = 0
    fallidos = 0
    archivos_procesados = []
    
    for archivo in archivos_txt:
        tabla_resultado, archivo_salida, exito = procesar_archivo_precipitacion(archivo)
        
        if exito:
            exitosos += 1
            archivos_procesados.append({
                'archivo_entrada': os.path.basename(archivo),
                'archivo_salida': os.path.basename(archivo_salida),
                'años': f"{tabla_resultado.index.min()}-{tabla_resultado.index.max()}"
            })
        else:
            fallidos += 1
    
    print(f"\n{'='*60}")
    print(f"REPORTE FINAL DE PROCESAMIENTO")
    print(f"{'='*60}")
    print(f"Archivos procesados exitosamente: {exitosos}")
    print(f"Archivos con errores: {fallidos}")
    print(f"Total de archivos: {len(archivos_txt)}")
    
    if archivos_procesados:
        print(f"\nARCHIVOS GENERADOS:")
        for i, info in enumerate(archivos_procesados, 1):
            print(f"   {i}. {info['archivo_entrada']} -> {info['archivo_salida']} ({info['años']})")
        
        print(f"\nUbicación de archivos: {ruta_carpeta}")
    
    return archivos_procesados

def procesar_todas_subcarpetas(ruta_carpeta_principal):
    """
    Procesa todas las subcarpetas dentro de una carpeta principal
    """
    print(f"EXPLORANDO CARPETA PRINCIPAL: {ruta_carpeta_principal}")
    print("=" * 70)
    
    if not os.path.exists(ruta_carpeta_principal):
        print(f"Error: La carpeta principal no existe: {ruta_carpeta_principal}")
        return
    
    subcarpetas = [f for f in os.listdir(ruta_carpeta_principal) 
                   if os.path.isdir(os.path.join(ruta_carpeta_principal, f))]
    
    if not subcarpetas:
        print(f"No se encontraron subcarpetas en: {ruta_carpeta_principal}")
        return
    
    print(f"Subcarpetas encontradas: {len(subcarpetas)}")
    for i, subcarpeta in enumerate(subcarpetas, 1):
        ruta_completa = os.path.join(ruta_carpeta_principal, subcarpeta)
        archivos_txt = glob.glob(os.path.join(ruta_completa, "*.txt"))
        print(f"   {i}. {subcarpeta} -> {len(archivos_txt)} archivos .txt")
    
    total_exitosos = 0
    total_fallidos = 0
    total_archivos = 0
    reporte_global = []
    
    for i, subcarpeta in enumerate(subcarpetas, 1):
        ruta_subcarpeta = os.path.join(ruta_carpeta_principal, subcarpeta)
        
        print(f"\nPROCESANDO SUBCARPETA {i}/{len(subcarpetas)}: {subcarpeta}")
        print("=" * 70)
        
        archivos_txt = glob.glob(os.path.join(ruta_subcarpeta, "*.txt"))
        
        if not archivos_txt:
            print(f"   No se encontraron archivos .txt en: {subcarpeta}")
            continue
        
        archivos_procesados = procesar_carpeta_completa(ruta_subcarpeta)
        
        if archivos_procesados:
            total_exitosos += len(archivos_procesados)
            total_archivos += len(archivos_txt)
            
            reporte_global.append({
                'subcarpeta': subcarpeta,
                'archivos_procesados': len(archivos_procesados),
                'total_archivos': len(archivos_txt),
                'detalles': archivos_procesados
            })
        else:
            total_fallidos += len(archivos_txt) if archivos_txt else 0
            total_archivos += len(archivos_txt) if archivos_txt else 0
    

    print(f"Subcarpetas procesadas: {len(subcarpetas)}")
    print(f"Total archivos exitosos: {total_exitosos}")
    print(f"Total archivos fallidos: {total_fallidos}")
    print(f"Total archivos encontrados: {total_archivos}")
    
    if reporte_global:
        print(f"\nRESUMEN POR SUBCARPETA:")
        for info in reporte_global:
            print(f"\n{info['subcarpeta']}:")
            print(f"   Exitosos: {info['archivos_procesados']}/{info['total_archivos']}")
            for detalle in info['detalles']:
                print(f"   - {detalle['archivo_entrada']} -> {detalle['archivo_salida']} ({detalle['años']})")
    
    return reporte_global

def crear_estadisticas(tabla_pivote):
    """
    Crea estadísticas básicas de la tabla de precipitación
    """
    estadisticas = pd.DataFrame()
    
    for columna in tabla_pivote.columns:
        datos = tabla_pivote[columna].dropna()
        if len(datos) > 0:
            estadisticas[columna] = [
                datos.mean(),      # Promedio
                datos.median(),    # Mediana
                datos.max(),       # Máximo
                datos.min(),       # Mínimo
                datos.std(),       # Desviación estándar
                len(datos)         # Número de años con datos
            ]
    
    estadisticas.index = ['Promedio (mm)', 'Mediana (mm)', 'Máximo (mm)', 
                         'Mínimo (mm)', 'Desv_Estándar (mm)', 'Años_con_datos']
    
    return estadisticas

def procesamiento_completo(ruta_carpeta_principal=None):
    """
    Procesamiento completo de todas las subcarpetas de Data_SEMANHI
    """
    if ruta_carpeta_principal is None:
        ruta_carpeta_principal = r"C:\Users\Master\Desktop\GitHub\WaterKu\data\Data_SEMANHI"
    
    print("PROCESAMIENTO COMPLETO DE DATA_SEMANHI")
    print("=" * 50)
    print(f"Ruta principal: {ruta_carpeta_principal}")
    
    if not os.path.exists(ruta_carpeta_principal):
        print(f"Error: La carpeta no existe: {ruta_carpeta_principal}")
        return []
    
    subcarpetas = [f for f in os.listdir(ruta_carpeta_principal) 
                   if os.path.isdir(os.path.join(ruta_carpeta_principal, f))]
    
    print(f"Subcarpetas encontradas: {len(subcarpetas)}")
    
    total_archivos = 0
    archivos_salida = []
    
    for subcarpeta in subcarpetas:
        ruta_sub = os.path.join(ruta_carpeta_principal, subcarpeta)
        archivos = glob.glob(os.path.join(ruta_sub, "*.txt"))
        total_archivos += len(archivos)
        print(f"  {subcarpeta} -> {len(archivos)} archivos .txt")
        
        for archivo in archivos:
            nombre_base = os.path.splitext(os.path.basename(archivo))[0]
            archivo_excel = os.path.join(ruta_sub, f"{nombre_base}_precip_max_24h.xlsx")
            archivos_salida.append(archivo_excel)
    
    print(f"Total archivos a procesar: {total_archivos}")
    print(f"Archivos Excel a generar: {len(archivos_salida)}")
    print("\nIniciando procesamiento...")
    
    procesar_todas_subcarpetas(ruta_carpeta_principal)
    
    return archivos_salida

def main():
    """
    Función principal para Snakemake
    """
    try:
        with open(snakemake.input.basin_config) as f:
            basin_config = yaml.safe_load(f)
        
        ruta_datos = getattr(snakemake.params, 'data_path', 'data/Data_SEMANHI')
        ruta_carpeta_principal = os.path.join(os.getcwd(), ruta_datos)
        
        print("EJECUTANDO PROCESAMIENTO PRECIPITACIÓN")
        print(f"Configuración cargada desde: {snakemake.input.basin_config}")
        print(f"Procesando datos en: {ruta_carpeta_principal}")
        
        archivos_generados = procesamiento_completo(ruta_carpeta_principal)
        
        output_file = snakemake.output[0]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write("PROCESAMIENTO DE PRECIPITACIÓN COMPLETADO\n")
            f.write(f"Fecha: {pd.Timestamp.now()}\n")
            f.write(f"Archivos procesados: {len(archivos_generados)}\n")
            f.write(f"Ruta procesada: {ruta_carpeta_principal}\n\n")
            f.write("ARCHIVOS GENERADOS:\n")
            for archivo in archivos_generados:
                f.write(f"  {archivo}\n")
        
        print(f"Reporte guardado en: {output_file}")
        
    except Exception as e:
        print(f"Error en configuración: {e}")
        raise e

if __name__ == "__main__":
    print("PROCESADOR DE PRECIPITACIÓN MÁXIMA 24H - SENAHMI")
    print("=" * 60)
    
    main()
    
    print("\nProcesamiento completado")
