import pandas as pd
import numpy as np
import re
import yaml
import os
from utils import should_use_excel, load_cuenca_config

def get_cuenca_from_path(path):
    """Extraer nombre de cuenca del path de salida o entrada"""
    parts = path.replace('\\', '/').split('/')
    if 'results' in parts:
        idx = parts.index('results')
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None

def leer_area_cuenca(parametros_path, cuenca_config):
    """
    Lee el área de la cuenca, usando el flag usar_parametros_excel para decidir la fuente
    """
    usar_excel = should_use_excel(parametros_path, cuenca_config)

    if usar_excel:
        try:
            df_parametros = pd.read_excel(parametros_path, index_col=0)
            area_km2 = df_parametros.loc["Área (Km²)", "Valor"]
            area_ha = area_km2 * 100
            if "Tiempo de concentración (Kirpich) (min)" in df_parametros.index:
                tc_min = df_parametros.loc["Tiempo de concentración (Kirpich) (min)", "Valor"]
            else:
                tc_min = cuenca_config.get('tiempo_concentracion_min', 45.0)
            return area_ha, area_km2, tc_min
        except Exception as e:
            usar_excel = False

    if not usar_excel:
        area_km2 = cuenca_config.get('area_cuenca_km2', 2.5)
        area_ha = cuenca_config.get('area_cuenca_ha', area_km2 * 100)
        tc_min = cuenca_config.get('tiempo_concentracion_min', 45.0)
        return area_ha, area_km2, tc_min

def leer_intensidad_idf(idf_excel_path, tr_objetivo, tc_min):
    try:
        df_idf = pd.read_excel(idf_excel_path)
        fila_tr = df_idf[df_idf["T (años)"] == tr_objetivo]
        if fila_tr.empty:
            return None
        dur_cols = [col for col in df_idf.columns if col.startswith("I_")]
        duraciones_min = []
        for col in dur_cols:
            parte = col.split('_')[1]
            if 'h' in parte:
                horas = parte.split('h')[0]
            else:
                horas = parte
            try:
                h = float(horas)
                duraciones_min.append(h * 60)
            except Exception:
                duraciones_min.append(np.nan)
        dur_cols_filtradas = []
        duraciones_min_filtradas = []
        for col, dmin in zip(dur_cols, duraciones_min):
            if not np.isnan(dmin):
                dur_cols_filtradas.append(col)
                duraciones_min_filtradas.append(dmin)
        if tc_min in duraciones_min_filtradas:
            idx = duraciones_min_filtradas.index(tc_min)
            intensidad = float(fila_tr.iloc[0][dur_cols_filtradas[idx]])
            return intensidad
        else:
            x = np.array(duraciones_min_filtradas)
            y = np.array([float(fila_tr.iloc[0][col]) for col in dur_cols_filtradas])
            intensidad_interp = np.interp(tc_min, x, y)
            return intensidad_interp
    except Exception as e:
        return None

def calcular_caudal_racional(area_ha, intensidad_mm_h, C):
    Q_ms = (C * intensidad_mm_h * area_ha) / 360
    return Q_ms

def decide_metodo(area_ha):
    if area_ha <= 1000:
        return 'racional'
    else:
        return 'otro'

# ---------------------------------------------------------------------------
# Funciones para cuencas de proyecto (cuenca_ya_delimitada=true)
# ---------------------------------------------------------------------------

def _tc_scs(longitud_km, pendiente_pct, cn):
    """Tiempo de concentración SCS (misma fórmula que hidrograma_scs.py)."""
    L_m = longitud_km * 1000
    return (4.3611 * (L_m ** 0.8) * ((1000 / cn - 9) ** 0.7)) / (1900 * (pendiente_pct ** 0.5))


def _leer_idf_proyecto(idf_path, Tc_h, periodos):
    """Lee P24h e interpola I(Tc_h) para cada TR desde IDF_resultados.xlsx."""
    df = pd.read_excel(idf_path)
    dur_cols = {}
    for col in df.columns:
        m = re.match(r'I_([\d.]+)h', str(col))
        if m:
            dur_cols[float(m.group(1))] = col
    durations = sorted(dur_cols.keys())

    result = {}
    for tr in periodos:
        row = df[df["T (años)"] == tr]
        if row.empty:
            result[tr] = {"I": None, "P24h": None}
            continue
        P24h = float(row["P24 (mm)"].values[0])
        vals = {d: float(row[dur_cols[d]].values[0]) for d in durations}
        if Tc_h <= durations[0]:
            I = vals[durations[0]]
        elif Tc_h >= durations[-1]:
            I = vals[durations[-1]]
        else:
            for i in range(len(durations) - 1):
                d1, d2 = durations[i], durations[i + 1]
                if d1 <= Tc_h <= d2:
                    t = (Tc_h - d1) / (d2 - d1)
                    I = vals[d1] + t * (vals[d2] - vals[d1])
                    break
        result[tr] = {"I": round(I, 2), "P24h": round(P24h, 2)}
    return result


def _main_proyecto():
    """Modo cuencas de proyecto: múltiples TRs, salida en formato Resumen Caudales."""
    par_path  = str(snakemake.input.parametros)
    idf_path  = str(snakemake.input.idf)
    output_xl = str(snakemake.output[0])
    periodos  = list(snakemake.params.periodos_retorno)
    basins_cfg = snakemake.params.basins_cfg
    cuenca_id  = getattr(snakemake.wildcards, "cuenca", "")

    bcfg   = basins_cfg.get(cuenca_id, {})
    cn     = float(bcfg.get("cn_default", 75))
    C      = float(bcfg.get("coeficiente_escorrentia", 0.85))

    # Parámetros geomorfológicos
    df_par   = pd.read_excel(par_path, index_col=0)
    val_col  = df_par.columns[0]
    def _get(keys):
        for k in keys:
            for idx in df_par.index:
                if k.lower() in str(idx).lower():
                    try: return float(df_par.loc[idx, val_col])
                    except: pass
        return None
    area_km2 = _get(["Área (Km", "Area (Km"])
    area_ha  = _get(["Área (Ha", "Area (Ha"]) or (area_km2 * 100 if area_km2 else None)
    L_km     = _get(["Longitud", "longitud"])
    Sc_pct   = _get(["Pendiente", "pendiente"])

    Tc_h = _tc_scs(L_km, Sc_pct, cn)

    datos = _leer_idf_proyecto(idf_path, Tc_h, periodos)

    Q = {}
    for tr in periodos:
        d = datos.get(tr, {})
        I = d.get("I")
        Q[tr] = round(C * I * area_ha / 360, 3) if I is not None else None

    print(f"  Metodo Racional - Cuenca {cuenca_id}")
    print(f"    Tc = {Tc_h:.3f} h | A = {area_ha:.2f} ha | CN = {cn} | C = {C}")
    for tr in periodos:
        print(f"    Q{tr} = {Q[tr]} m3/s  (I={datos[tr]['I']} mm/h  P24h={datos[tr]['P24h']} mm)")

    os.makedirs(os.path.dirname(output_xl), exist_ok=True)
    rows = [("Período de Retorno (años)", "P24h (mm)", "Caudal Pico (m³/s)", "Intensidad I(Tc) (mm/h)")]
    for tr in periodos:
        d = datos.get(tr, {})
        rows.append((tr, d.get("P24h", ""), Q.get(tr, ""), d.get("I", "")))
    rows += [
        (None, None, None, None),
        ("Tc (horas)",  round(Tc_h, 3),  None, None),
        ("Área (km²)",  area_km2,         None, None),
        ("CN",          cn,               None, None),
    ]
    df_out = pd.DataFrame(rows)
    with pd.ExcelWriter(output_xl, engine="openpyxl") as writer:
        df_out.to_excel(writer, sheet_name="Resumen Caudales", index=False, header=False)
    print(f"  Salida: {output_xl}")


def main():
    # Modo cuencas de proyecto (periodos_retorno en params)
    if 'snakemake' in globals() and hasattr(snakemake.params, 'periodos_retorno'):
        _main_proyecto()
        return

    # Para Snakemake: obtener cuenca del path de salida
    if 'snakemake' in globals():
        cuenca_id = get_cuenca_from_path(str(snakemake.output[0]))
        cuenca_config, global_config = load_cuenca_config(cuenca_id)
        parametros_excel_path = str(snakemake.input.parametros)
        idf_excel_path = str(snakemake.input.idf)
        output_path = str(snakemake.output[0])
    else:
        # Para ejecución directa: usar configuración legacy
        with open('config/config.yaml', 'r', encoding='utf-8') as file:
            global_config = yaml.safe_load(file)
        cuenca_config = global_config  # fallback para compatibilidad
        parametros_excel_path = global_config['parametros_excel']
        idf_excel_path = global_config['idf_resultados']
        output_path = global_config['resultados_caudal_racional']

    C = cuenca_config.get('coeficiente_escorrentia', 0.85)
    tr_objetivo = cuenca_config.get('tr_metodo_racional', 100)
    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)

    try:
        area_ha, area_km2, tc_min = leer_area_cuenca(parametros_excel_path, cuenca_config)
        if area_ha is None:
            raise ValueError("No se pudo leer el área de la cuenca")

        if area_ha > 1000:
            datos_no_aplica = {
                "Parámetro": [
                    "Área de la cuenca (ha)",
                    "Límite método racional (ha)",
                    "Método aplicable",
                    "Observación"
                ],
                "Valor": [
                    f"{area_ha:.2f}",
                    "1000",
                    "NO",
                    "Se requiere otro método"
                ]
            }
            df_no_aplica = pd.DataFrame(datos_no_aplica)
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_no_aplica.to_excel(writer, sheet_name='Método Racional', index=False)
                worksheet = writer.sheets['Método Racional']
                worksheet.column_dimensions['A'].width = 30
                worksheet.column_dimensions['B'].width = 50
                from openpyxl.styles import Alignment
                for row in range(2, len(datos_no_aplica["Parámetro"]) + 2):
                    worksheet[f'B{row}'].alignment = Alignment(horizontal='right')
            return

        metodo = decide_metodo(area_ha)
        resultado = {
            "area_ha": area_ha,
            "area_km2": area_km2,
            "metodo": metodo,
            "Tr_años": tr_objetivo,
            "Duracion_min": tc_min,
            "Intensidad_mm_h": np.nan,
            "Coef_escorrentia": C,
            "Caudal_m3s": np.nan
        }

        if tc_min is None:
            raise ValueError("No se encontró el tiempo de concentración en el Excel de parámetros")

        intensidad = leer_intensidad_idf(idf_excel_path, tr_objetivo, tc_min)
        if intensidad is None:
            raise ValueError("No se pudo leer/interpolar la intensidad IDF")

        resultado["Intensidad_mm_h"] = intensidad
        Q_ms = calcular_caudal_racional(area_ha, intensidad, C)
        resultado["Caudal_m3s"] = Q_ms

        datos_excel = {
            "Parámetro": [
                "Área utilizada (ha)",
                "Período de retorno (años)",
                "Tiempo de concentración (min)",
                "Intensidad (mm/h)",
                "Coeficiente de escorrentía",
                "Caudal (m³/s)"
            ],
            "Valor": [
                f"{area_ha:.3f}",
                tr_objetivo,
                f"{tc_min:.2f}",
                f"{resultado['Intensidad_mm_h']:.3f}",
                C,
                f"{resultado['Caudal_m3s']:.3f}"
            ]
        }
        df_resultado = pd.DataFrame(datos_excel)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_resultado.to_excel(writer, sheet_name='Método Racional', index=False)
            worksheet = writer.sheets['Método Racional']
            worksheet.column_dimensions['A'].width = 30
            worksheet.column_dimensions['B'].width = 20
            from openpyxl.styles import Alignment
            for row in range(2, len(datos_excel["Parámetro"]) + 2):
                worksheet[f'B{row}'].alignment = Alignment(horizontal='right')

    except Exception as e:
        datos_error = {
            "Parámetro": [
                "Estado",
                "Error"
            ],
            "Valor": [
                "ERROR EN CÁLCULO",
                str(e)
            ]
        }
        df_error = pd.DataFrame(datos_error)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df_error.to_excel(writer, sheet_name='Método Racional', index=False)
            worksheet = writer.sheets['Método Racional']
            worksheet.column_dimensions['A'].width = 30
            worksheet.column_dimensions['B'].width = 50
            from openpyxl.styles import Alignment
            for row in range(2, len(datos_error["Parámetro"]) + 2):
                worksheet[f'B{row}'].alignment = Alignment(horizontal='right')

if __name__ == "__main__":
    main()
