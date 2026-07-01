"""
WaterKu — Módulo MODFLOW 6 Lagoon-Aquifer Interaction
======================================================
Construye, ejecuta y post-procesa modelos de flujo subterráneo
(MODFLOW 6 / FloPy) para estudiar la interacción laguna–acuífero.

Soporta dos representaciones de la laguna:
  GHB  — General Head Boundary (carga fija, simple)
  LAK6 — Lake Package (nivel dinámico, realista)
  BOTH — Corre ambos y genera tabla comparativa

Entrada  : Excel (8 sheets) + YAML de configuración
Salida   : Excel con resultados + PNGs de contornos de carga
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import flopy
import flopy.plot
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class ModflowLagunaModel:
    """
    Generalización del análisis laguna-acuífero en MODFLOW 6 / FloPy.
    Todos los parámetros geométricos e hidráulicos se leen desde Excel.
    """

    def __init__(self, config_path: str | Path):
        config_path = Path(config_path)
        with config_path.open() as f:
            self.cfg = yaml.safe_load(f)

        # Resolver rutas relativas al directorio raíz del proyecto
        # (este script está en scripts/estructuras_hidraulicas/)
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.excel_path = self.project_root / self.cfg["input_excel"]
        self.output_dir = self.project_root / self.cfg["output_dir"]
        self.workspace_base = self.project_root / self.cfg["workspace_dir"]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_base.mkdir(parents=True, exist_ok=True)

        self.mf6_exe = self._resolve_mf6_exe()
        self.verbose = self.cfg.get("verbose", True)

    # ── Utilidades internas ────────────────────────────────────────────────

    def _resolve_mf6_exe(self) -> str:
        primary = self.cfg.get("mf6_exe", "")
        if primary and Path(primary).exists():
            return primary
        fallback = self.cfg.get("mf6_exe_fallback", "mf6")
        if shutil.which(fallback):
            return fallback
        raise FileNotFoundError(
            f"No se encontró el ejecutable MODFLOW 6.\n"
            f"  Ruta configurada : {primary}\n"
            f"  Fallback en PATH : {fallback}\n"
            f"Descarga: https://github.com/MODFLOW-USGS/modflow6/releases"
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[modflow_laguna] {msg}")

    # ── Carga de Excel ─────────────────────────────────────────────────────

    def _load_excel(self) -> dict[str, pd.DataFrame]:
        self._log(f"Leyendo Excel: {self.excel_path}")
        sheets = pd.read_excel(self.excel_path, sheet_name=None)
        return sheets

    def _parse_configuracion(self, sheets: dict) -> dict:
        df = sheets["Configuracion"].copy()
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.set_index("parametro")["valor"]
        conf = {}
        for k, v in df.items():
            k = str(k).strip()
            if isinstance(v, str):
                v_upper = v.strip().upper()
                if v_upper == "TRUE":
                    v = True
                elif v_upper == "FALSE":
                    v = False
                else:
                    try:
                        v = int(v)
                    except (ValueError, TypeError):
                        try:
                            v = float(v)
                        except (ValueError, TypeError):
                            v = v.strip()
            conf[k] = v
        return conf

    # ── Construcción de arrays de grilla ──────────────────────────────────

    def _build_grid_arrays(
        self, conf: dict, sheets: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve (top, botm, idomain_base) como arrays numpy."""
        nlay = int(conf["nlay"])
        nrow = int(conf["nrow"])
        ncol = int(conf["ncol"])
        fmt = self.cfg.get("elevaciones_formato", "constante")

        if fmt == "constante":
            top, botm = self._build_grid_arrays_constante(sheets, nlay, nrow, ncol)
        else:
            top, botm = self._build_grid_arrays_variable(sheets, nlay, nrow, ncol)

        idomain = np.ones((nlay, nrow, ncol), dtype=int)
        return top, botm, idomain

    def _build_grid_arrays_constante(
        self, sheets: dict, nlay: int, nrow: int, ncol: int
    ) -> tuple[np.ndarray, np.ndarray]:
        df = sheets["Elevaciones"].copy()
        df.columns = [c.strip().lower() for c in df.columns]
        df["tipo"] = df["tipo"].str.upper()

        top_val = float(df[df["tipo"] == "TOP"]["elevacion_m"].iloc[0])
        top = np.full((nrow, ncol), top_val, dtype=float)

        botm = np.zeros((nlay, nrow, ncol), dtype=float)
        for _, row in df[df["tipo"] == "BOTM"].iterrows():
            layer = int(row["layer"]) - 1
            val = float(row["elevacion_m"])
            botm[layer] = val

        return top, botm

    def _build_grid_arrays_variable(
        self, sheets: dict, nlay: int, nrow: int, ncol: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lee sheets TOP, BOTM_L1, BOTM_L2, ... como matrices nrow×ncol."""
        top = np.array(sheets["TOP"].values, dtype=float)
        botm = np.zeros((nlay, nrow, ncol), dtype=float)
        for lay in range(nlay):
            sheet_name = f"BOTM_L{lay+1}"
            botm[lay] = np.array(sheets[sheet_name].values, dtype=float)
        return top, botm

    def _build_strt_arrays(self, conf: dict, sheets: dict):
        """
        Devuelve array STRT (nlay, nrow, ncol) o escalar float.
        strt_formato="variable" lee sheets STRT_L1, STRT_L2, STRT_L3.
        strt_formato="constante" (default) usa initial_head escalar.
        """
        fmt = self.cfg.get("strt_formato", "constante")
        nlay = int(conf["nlay"])
        nrow = int(conf["nrow"])
        ncol = int(conf["ncol"])
        if fmt == "variable":
            strt = np.zeros((nlay, nrow, ncol), dtype=float)
            for lay in range(nlay):
                sheet_name = f"STRT_L{lay+1}"
                strt[lay] = np.array(sheets[sheet_name].values, dtype=float)
            return strt
        return float(conf.get("initial_head", 30.0))

    # ── Laguna: idomain + connectiondata para LAK6 ────────────────────────

    def _build_lagoon_idomain_connections(
        self, idomain_base: np.ndarray, laguna_df: pd.DataFrame, conf: dict
    ) -> tuple[np.ndarray, int, list]:
        idomain = idomain_base.copy()
        connectiondata = []
        iconn = 0

        for _, row in laguna_df.iterrows():
            r = int(row["fila_1base"]) - 1
            c = int(row["columna_1base"]) - 1
            lay = int(row["layer_1base"]) - 1
            leakance = float(row["leakance_md"])
            idomain[lay, r, c] = 0
            connectiondata.append(
                [0, iconn, (lay + 1, r, c), "vertical", leakance, 0.0, 0.0, 0.0, 0.0]
            )
            iconn += 1

        return idomain, iconn, connectiondata

    # ── Construcción del modelo FloPy ─────────────────────────────────────

    def build_model(
        self,
        approach: str,
        conf: dict,
        sheets: dict,
        workspace: Path,
    ):
        """
        Construye MFSimulation + todos los paquetes.
        approach: "GHB" o "LAK6"
        """
        nlay = int(conf["nlay"])
        nrow = int(conf["nrow"])
        ncol = int(conf["ncol"])
        delr = float(conf["delr_m"])
        delc = float(conf["delc_m"])
        model_name = str(conf.get("model_name", "modflow_laguna"))

        top, botm, idomain_base = self._build_grid_arrays(conf, sheets)

        # Datos de contorno
        laguna_df = sheets["Laguna_Celdas"].copy()
        laguna_df.columns = [c.strip().lower() for c in laguna_df.columns]

        ghb_regional_df = sheets["GHB_Regional"].copy()
        ghb_regional_df.columns = [c.strip().lower() for c in ghb_regional_df.columns]

        riv_df = sheets["RIV"].copy()
        riv_df.columns = [c.strip().lower() for c in riv_df.columns]

        rch_df = sheets["RCH"].copy()
        rch_df.columns = [c.strip().lower() for c in rch_df.columns]

        piezo_df = sheets["Piezometros"].copy()
        piezo_df.columns = [c.strip().lower() for c in piezo_df.columns]

        # Conjunto de celdas de laguna (0-based) para excluir de RCH
        lake_cells = set()
        for _, r in laguna_df.iterrows():
            lake_cells.add((int(r["layer_1base"]) - 1,
                            int(r["fila_1base"]) - 1,
                            int(r["columna_1base"]) - 1))

        # ── Simulación ─────────────────────────────────────────────────────
        workspace.mkdir(parents=True, exist_ok=True)
        sim = flopy.mf6.MFSimulation(
            sim_name="mfsim", sim_ws=str(workspace), exe_name=self.mf6_exe
        )

        # Tiempo
        perlen = float(conf.get("perlen", 1.0))
        nstp = int(conf.get("nstp", 1))
        tsmult = float(conf.get("tsmult", 1.0))
        time_units = str(conf.get("time_units", "SECONDS"))
        flopy.mf6.ModflowTdis(
            sim,
            time_units=time_units,
            nper=int(conf.get("nper", 1)),
            perioddata=[(perlen, nstp, tsmult)],
        )

        # Solver IMS
        complexity = str(conf.get("solver_complexity", "SIMPLE"))
        ims = flopy.mf6.ModflowIms(
            sim,
            complexity=complexity,
            outer_hclose=float(conf.get("outer_hclose", 1e-3)),
            outer_maximum=int(conf.get("outer_maximum", 25)),
            inner_maximum=int(conf.get("inner_maximum", 50)),
            inner_hclose=float(conf.get("inner_hclose", 1e-3)),
            rcloserecord=[1.0e-1, "STRICT"],
            linear_acceleration="CG",
        )

        # Modelo GWF
        length_units = str(conf.get("length_units", "METERS"))
        gwf = flopy.mf6.ModflowGwf(
            sim, modelname=model_name, save_flows=True
        )
        sim.register_ims_package(ims, [gwf.name])

        # DIS — diferente idomain según enfoque
        if approach == "LAK6":
            idomain, nconn, connectiondata = self._build_lagoon_idomain_connections(
                idomain_base, laguna_df, conf
            )
        else:
            idomain = idomain_base.copy()

        flopy.mf6.ModflowGwfdis(
            gwf,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            delr=delr,
            delc=delc,
            top=top,
            botm=botm,
            idomain=idomain,
            xorigin=float(conf.get("xorigin", 0.0)),
            yorigin=float(conf.get("yorigin", 0.0)),
            length_units=length_units,
        )

        # IC
        strt = self._build_strt_arrays(conf, sheets)
        flopy.mf6.ModflowGwfic(gwf, strt=strt)

        # NPF
        capas_df = sheets["Capas"].copy()
        capas_df.columns = [c.strip().lower() for c in capas_df.columns]
        capas_df = capas_df.sort_values("layer")
        k_vals = capas_df["k_md"].values.astype(float)
        k33_vals = capas_df["k33_md"].values.astype(float)
        k22_vals = capas_df["k22_md"].values.astype(float)
        icelltype = capas_df["icelltype"].values.astype(int)

        flopy.mf6.ModflowGwfnpf(
            gwf,
            icelltype=icelltype,
            k=k_vals,
            k33=k33_vals,
            k22=k22_vals,
            save_specific_discharge=True,
        )

        # RIV
        riv_spd = []
        for _, r in riv_df.iterrows():
            cellid = (int(r["layer_1base"]) - 1, int(r["fila_1base"]) - 1, int(r["columna_1base"]) - 1)
            riv_spd.append([cellid, float(r["stage_m"]), float(r["cond_m2d"]), float(r["rbot_m"])])
        flopy.mf6.ModflowGwfriv(gwf, boundnames=False, stress_period_data={0: riv_spd})

        # GHB — regional flow (siempre) + laguna (solo en modo GHB)
        ghb_spd = []
        for _, r in ghb_regional_df.iterrows():
            cellid = (int(r["layer_1base"]) - 1, int(r["fila_1base"]) - 1, int(r["columna_1base"]) - 1)
            ghb_spd.append([cellid, float(r["bhead_m"]), float(r["cond_m2d"])])

        if approach == "GHB":
            lagoon_head = float(conf.get("lagoon_initial_stage", 29.5))
            for _, r in laguna_df.iterrows():
                cellid = (int(r["layer_1base"]) - 1, int(r["fila_1base"]) - 1, int(r["columna_1base"]) - 1)
                cond = float(r["leakance_md"]) * float(conf.get("delr_m", 50.0)) * float(conf.get("delc_m", 50.0))
                ghb_spd.append([cellid, lagoon_head, cond])

        flopy.mf6.ModflowGwfghb(gwf, boundnames=False, stress_period_data={0: ghb_spd})

        # RCH
        rch_uniforme = self.cfg.get("rch_uniforme", True)
        rch_spd = []
        if rch_uniforme and len(rch_df) == 1 and int(rch_df.iloc[0]["fila_1base"]) == -1:
            rate = float(rch_df.iloc[0]["recarga_m_s"])
            for lay in range(1):  # recarga solo en capa superior (lay=0)
                for ri in range(nrow):
                    for ci in range(ncol):
                        cellid = (lay, ri, ci)
                        if approach == "LAK6" and cellid in lake_cells:
                            continue
                        rch_spd.append([cellid, rate])
        else:
            for _, r in rch_df.iterrows():
                cellid = (int(r["layer_1base"]) - 1, int(r["fila_1base"]) - 1, int(r["columna_1base"]) - 1)
                if approach == "LAK6" and cellid in lake_cells:
                    continue
                rch_spd.append([cellid, float(r["recarga_m_s"])])

        flopy.mf6.ModflowGwfrch(gwf, boundnames=False, stress_period_data={0: rch_spd})

        # LAK6 (solo si approach == "LAK6")
        if approach == "LAK6":
            initial_stage = float(conf.get("lagoon_initial_stage", 29.5))
            rainfall = float(conf.get("lagoon_rainfall_rate", 0.0))
            evap = float(conf.get("lagoon_evaporation", 0.0))
            runoff = float(conf.get("lagoon_runoff", 0.0))

            packagedata = [[0, initial_stage, nconn, "laguna"]]
            perioddata = [
                [0, "status", "active"],
                [0, "rainfall", rainfall],
                [0, "evaporation", evap],
                [0, "runoff", runoff],
            ]
            flopy.mf6.ModflowGwflak(
                gwf,
                pname="lak",
                boundnames=True,
                print_stage=True,
                print_flows=True,
                save_flows=True,
                stage_filerecord=f"{model_name}.lak.stage.bin",
                budget_filerecord=f"{model_name}.lak.bud.bin",
                nlakes=1,
                noutlets=0,
                packagedata=packagedata,
                connectiondata=connectiondata,
                perioddata=perioddata,
            )

        # OC — output control
        flopy.mf6.ModflowGwfoc(
            gwf,
            head_filerecord=f"{model_name}.bhd",
            budget_filerecord=f"{model_name}.cbc",
            saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        )

        # OBS — piezómetros
        obs_list = []
        for _, r in piezo_df.iterrows():
            name = str(r["nombre"]).strip()
            cellid = (int(r["layer_1base"]) - 1, int(r["fila_1base"]) - 1, int(r["columna_1base"]) - 1)
            obs_list.append((name, "HEAD", cellid))

        obs_filename = f"{model_name}.ob_heads.csv"
        flopy.mf6.ModflowUtlobs(
            gwf,
            filename=f"{model_name}.ob_gw",
            continuous={obs_filename: obs_list},
        )

        return sim, gwf

    # ── Ejecución ──────────────────────────────────────────────────────────

    def run_simulation(self, sim) -> tuple[bool, list]:
        sim.write_simulation()
        success, buff = sim.run_simulation()
        if not success:
            last_lines = buff[-30:] if buff else []
            raise RuntimeError(
                f"MODFLOW 6 no convergió.\nÚltimas líneas del log:\n" + "\n".join(last_lines)
            )
        return success, buff

    # ── Extracción de resultados ───────────────────────────────────────────

    def extract_results(self, gwf, approach: str, workspace: Path, conf: dict) -> dict:
        model_name = str(conf.get("model_name", "modflow_laguna"))
        heads = gwf.output.head().get_data()

        # Budget
        budget_terms = {}
        try:
            cbc_path = workspace / f"{model_name}.cbc"
            if cbc_path.exists():
                cbf = flopy.utils.CellBudgetFile(str(cbc_path), precision="double")
                for term in cbf.get_unique_record_names():
                    try:
                        data = cbf.get_data(text=term)
                        if data:
                            total = float(np.sum([d for d in data[-1].flatten() if d > 0]))
                            budget_terms[term.strip()] = total
                    except Exception:
                        pass
        except Exception:
            pass

        # Lake stage (LAK6 only)
        lake_stage = None
        if approach == "LAK6":
            stage_path = workspace / f"{model_name}.lak.stage.bin"
            if stage_path.exists():
                try:
                    sf = flopy.utils.HeadFile(str(stage_path), text="STAGE")
                    lake_stage = float(sf.get_data()[0, 0, 0])
                except Exception:
                    pass

        # Piezómetros — leer CSV generado por MODFLOW
        piezometers = {}
        obs_csv = workspace / f"{model_name}.ob_heads.csv"
        if obs_csv.exists():
            df_obs = pd.read_csv(obs_csv)
            last_row = df_obs.iloc[-1]
            for col in df_obs.columns[1:]:
                piezometers[col.strip()] = float(last_row[col])

        return {
            "heads": heads,
            "budget_terms": budget_terms,
            "lake_stage": lake_stage,
            "piezometers": piezometers,
        }

    # ── Modo BOTH ──────────────────────────────────────────────────────────

    def compare_approaches(
        self, conf: dict, sheets: dict
    ) -> tuple[dict, dict, object, object]:
        results = {}
        gwfs = {}
        for approach in ("GHB", "LAK6"):
            workspace = self.workspace_base / approach
            self._log(f"Construyendo modelo [{approach}] en {workspace}")
            sim, gwf = self.build_model(approach, conf, sheets, workspace)
            self._log(f"Corriendo simulación [{approach}]...")
            self.run_simulation(sim)
            self._log(f"Extrayendo resultados [{approach}]...")
            results[approach] = self.extract_results(gwf, approach, workspace, conf)
            gwfs[approach] = gwf
        return results, gwfs

    # ── Exportación ────────────────────────────────────────────────────────

    def export_results(
        self,
        results: dict,
        approach: str,
        gwf_map: dict,
        conf: dict,
        sheets: dict,
        t_elapsed: float = 0.0,
    ) -> None:
        output_excel = self.output_dir / self.cfg.get("output_excel_filename", "modflow_laguna_resultados.xlsx")
        nlay = int(conf["nlay"])
        model_name = str(conf.get("model_name", "modflow_laguna"))

        # ── Excel ──────────────────────────────────────────────────────────
        with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
            # --- Resumen ---
            if approach == "BOTH":
                res_ghb = results["GHB"]
                res_lak = results["LAK6"]
                stage_ghb = float(conf.get("lagoon_initial_stage", 29.5))
                stage_lak = res_lak.get("lake_stage") or float("nan")
                diff_stage = stage_lak - stage_ghb
                summary = {
                    "campo": ["Modelo", "Enfoque_laguna", "Nro_capas", "Nro_filas", "Nro_columnas",
                              "Celda_m", "Dominio_m", "Stage_laguna_GHB_m", "Stage_laguna_LAK6_m",
                              "Diferencia_stage_m", "Tiempo_simulacion_s"],
                    "valor": [model_name, "BOTH", conf["nlay"], conf["nrow"], conf["ncol"],
                              conf["delr_m"],
                              f"{int(conf['nrow'])*float(conf['delr_m']):.0f} x {int(conf['ncol'])*float(conf['delc_m']):.0f}",
                              f"{stage_ghb:.3f}", f"{stage_lak:.3f}", f"{diff_stage:.4f}", f"{t_elapsed:.1f}"],
                }
            else:
                res = results[approach]
                stage = float(conf.get("lagoon_initial_stage", 29.5)) if approach == "GHB" else (res.get("lake_stage") or float("nan"))
                summary = {
                    "campo": ["Modelo", "Enfoque_laguna", "Nro_capas", "Nro_filas", "Nro_columnas",
                              "Celda_m", "Dominio_m", "Stage_laguna_m", "Convergencia", "Tiempo_simulacion_s"],
                    "valor": [model_name, approach, conf["nlay"], conf["nrow"], conf["ncol"],
                              conf["delr_m"],
                              f"{int(conf['nrow'])*float(conf['delr_m']):.0f} x {int(conf['ncol'])*float(conf['delc_m']):.0f}",
                              f"{stage:.3f}", "SI", f"{t_elapsed:.1f}"],
                }
            pd.DataFrame(summary).to_excel(writer, sheet_name="Resumen", index=False)

            # --- Cabezas por capa y enfoque ---
            for app_name, res in results.items():
                heads = res["heads"]
                for lay in range(nlay):
                    sheet_name = f"Cabezas_{app_name}_L{lay+1}"[:31]
                    df_h = pd.DataFrame(heads[lay])
                    df_h.index = [f"Fila_{i+1}" for i in range(df_h.shape[0])]
                    df_h.columns = [f"Col_{j+1}" for j in range(df_h.shape[1])]
                    df_h.to_excel(writer, sheet_name=sheet_name)

            # --- Piezómetros ---
            piezo_df_orig = sheets["Piezometros"].copy()
            piezo_df_orig.columns = [c.strip().lower() for c in piezo_df_orig.columns]
            piezo_rows = []
            for _, row in piezo_df_orig.iterrows():
                name = str(row["nombre"]).strip()
                entry = {
                    "piezometro": name,
                    "layer": int(row["layer_1base"]),
                    "fila": int(row["fila_1base"]),
                    "columna": int(row["columna_1base"]),
                }
                for app_name, res in results.items():
                    # MODFLOW outputs obs names in uppercase; match case-insensitively
                    piezo_map = {k.upper(): v for k, v in res["piezometers"].items()}
                    entry[f"cabeza_{app_name}_m"] = piezo_map.get(name.upper(), float("nan"))
                piezo_rows.append(entry)
            df_piezo = pd.DataFrame(piezo_rows)
            if approach == "BOTH" and "cabeza_GHB_m" in df_piezo.columns and "cabeza_LAK6_m" in df_piezo.columns:
                df_piezo["diferencia_m"] = df_piezo["cabeza_LAK6_m"] - df_piezo["cabeza_GHB_m"]
            df_piezo.to_excel(writer, sheet_name="Piezometros", index=False)

            # --- Presupuesto de agua ---
            budget_rows = []
            for app_name, res in results.items():
                for term, val in res["budget_terms"].items():
                    budget_rows.append({"componente": term, "enfoque": app_name, "caudal_m3_d": val})
            if budget_rows:
                df_budget = pd.DataFrame(budget_rows).pivot_table(
                    index="componente", columns="enfoque", values="caudal_m3_d", aggfunc="first"
                ).reset_index()
                df_budget.to_excel(writer, sheet_name="Presupuesto_Agua", index=False)

            # --- Comparación (solo si BOTH) ---
            if approach == "BOTH":
                comp_rows = []
                stage_ghb = float(conf.get("lagoon_initial_stage", 29.5))
                stage_lak = results["LAK6"].get("lake_stage") or float("nan")
                comp_rows.append({"variable": "Stage_laguna_m", "GHB": stage_ghb, "LAK6": stage_lak, "diferencia": stage_lak - stage_ghb})
                for row in piezo_rows:
                    name = row["piezometro"]
                    v_ghb = row.get("cabeza_GHB_m", float("nan"))
                    v_lak = row.get("cabeza_LAK6_m", float("nan"))
                    comp_rows.append({"variable": f"Cabeza_{name}_m", "GHB": v_ghb, "LAK6": v_lak, "diferencia": v_lak - v_ghb})
                pd.DataFrame(comp_rows).to_excel(writer, sheet_name="Comparacion", index=False)

        self._log(f"Excel guardado: {output_excel}")

        # ── PNGs ───────────────────────────────────────────────────────────
        plot_cfg = self.cfg.get("plots", {})
        if plot_cfg.get("head_contour", {}).get("enabled", True):
            layer_idx = int(plot_cfg.get("head_contour", {}).get("layer", 0))
            n_levels = int(plot_cfg.get("head_contour", {}).get("n_contour_levels", 15))
            dpi = int(plot_cfg.get("head_contour", {}).get("dpi", 150))
            cmap = plot_cfg.get("head_contour", {}).get("colormap", "viridis")
            figsize = tuple(plot_cfg.get("head_contour", {}).get("figsize", [7, 7]))

            for app_name, res in results.items():
                png_key = f"output_png_{app_name.lower()}"
                png_name = self.cfg.get(png_key, f"cabezas_{app_name.lower()}.png")
                out_png = self.output_dir / png_name
                gwf = gwf_map[app_name]
                self._plot_heads(
                    gwf=gwf,
                    heads=res["heads"],
                    piezo_df=piezo_df_orig,
                    approach=app_name,
                    layer=layer_idx,
                    n_levels=n_levels,
                    dpi=dpi,
                    cmap=cmap,
                    figsize=figsize,
                    out_path=out_png,
                )

            # Comparación side-by-side
            if approach == "BOTH" and plot_cfg.get("comparison", {}).get("enabled", True) and len(results) == 2:
                dpi_cmp = int(plot_cfg.get("comparison", {}).get("dpi", 150))
                figsize_cmp = tuple(plot_cfg.get("comparison", {}).get("figsize", [14, 7]))
                out_cmp = self.output_dir / self.cfg.get("output_png_comparison", "comparacion_ghb_lak6.png")
                self._plot_comparison(
                    gwf_ghb=gwf_map["GHB"],
                    gwf_lak=gwf_map["LAK6"],
                    heads_ghb=results["GHB"]["heads"],
                    heads_lak=results["LAK6"]["heads"],
                    piezo_df=piezo_df_orig,
                    layer=layer_idx,
                    n_levels=n_levels,
                    dpi=dpi_cmp,
                    figsize=figsize_cmp,
                    out_path=out_cmp,
                )

    # ── Plots ──────────────────────────────────────────────────────────────

    def _add_piezometers(self, ax, gwf, piezo_df: pd.DataFrame) -> None:
        xc, yc = gwf.modelgrid.xyzcellcenters[0], gwf.modelgrid.xyzcellcenters[1]
        for _, row in piezo_df.iterrows():
            ri = int(row["fila_1base"]) - 1
            ci = int(row["columna_1base"]) - 1
            x, y = float(xc[ri, ci]), float(yc[ri, ci])
            ax.plot(x, y, "r^", markersize=8)
            ax.annotate(str(row["nombre"]), (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    def _plot_heads(
        self,
        gwf,
        heads: np.ndarray,
        piezo_df: pd.DataFrame,
        approach: str,
        layer: int,
        n_levels: int,
        dpi: int,
        cmap: str,
        figsize: tuple,
        out_path: Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize)
        pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=layer)
        pmv.plot_grid(linewidth=0.3, color="0.7")
        cs = pmv.contour_array(heads[layer], levels=n_levels, cmap=cmap)
        ax.clabel(cs, inline=True, fontsize=7)
        pmv.plot_ibound()
        self._add_piezometers(ax, gwf, piezo_df)

        lake_patch = matplotlib.patches.Patch(facecolor="black", label="Laguna (idomain=0)")
        piezo_marker = matplotlib.lines.Line2D(
            [0], [0], marker="^", color="r", linestyle="None", label="Piezómetro"
        )
        ax.legend(handles=[lake_patch, piezo_marker], loc="lower left", fontsize=8)
        ax.set_title(f"Cabezas simuladas — Capa {layer+1} [{approach}]")
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        self._log(f"PNG guardado: {out_path}")

    def _plot_comparison(
        self,
        gwf_ghb,
        gwf_lak,
        heads_ghb: np.ndarray,
        heads_lak: np.ndarray,
        piezo_df: pd.DataFrame,
        layer: int,
        n_levels: int,
        dpi: int,
        figsize: tuple,
        out_path: Path,
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, gwf, heads, title in [
            (axes[0], gwf_ghb, heads_ghb, "GHB (carga fija)"),
            (axes[1], gwf_lak, heads_lak, "LAK6 (nivel dinámico)"),
        ]:
            pmv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=layer)
            pmv.plot_grid(linewidth=0.3, color="0.7")
            cs = pmv.contour_array(heads[layer], levels=n_levels, cmap="viridis")
            ax.clabel(cs, inline=True, fontsize=7)
            pmv.plot_ibound()
            self._add_piezometers(ax, gwf, piezo_df)
            ax.set_title(f"Cabezas — Capa {layer+1} [{title}]")

        fig.suptitle("Comparación GHB vs LAK6", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        self._log(f"PNG comparación guardado: {out_path}")

    # ── Entry point ────────────────────────────────────────────────────────

    def run(self) -> None:
        t0 = time.time()
        sheets = self._load_excel()
        conf = self._parse_configuracion(sheets)

        # Enfoque: YAML override > Excel
        approach = self.cfg.get("lagoon_approach_override") or str(conf.get("lagoon_approach", "LAK6")).upper()
        self._log(f"Enfoque de laguna: {approach}")

        if approach == "BOTH":
            results, gwf_map = self.compare_approaches(conf, sheets)
        else:
            workspace = self.workspace_base / approach
            self._log(f"Construyendo modelo en {workspace}")
            sim, gwf = self.build_model(approach, conf, sheets, workspace)
            self._log("Corriendo simulación MODFLOW 6...")
            self.run_simulation(sim)
            self._log("Extrayendo resultados...")
            res = self.extract_results(gwf, approach, workspace, conf)
            results = {approach: res}
            gwf_map = {approach: gwf}

        t_elapsed = time.time() - t0
        self._log(f"Simulación completada en {t_elapsed:.1f} s")

        # Mostrar piezómetros en consola
        for app_name, res in results.items():
            print(f"\n=== Piezómetros [{app_name}] ===")
            for name, head in res["piezometers"].items():
                print(f"  {name}: {head:.4f} m")
            if res.get("lake_stage") is not None:
                print(f"  Stage laguna: {res['lake_stage']:.4f} m")

        self.export_results(results, approach, gwf_map, conf, sheets, t_elapsed)
        self._log("Módulo modflow_laguna completado.")


# ---------------------------------------------------------------------------
# Función de entrada para Snakemake / scripts externos
# ---------------------------------------------------------------------------

def process_modflow_laguna(config_path: str, output_dir: str | None = None) -> None:
    model = ModflowLagunaModel(config_path)
    if output_dir:
        model.output_dir = Path(output_dir)
        model.output_dir.mkdir(parents=True, exist_ok=True)
    model.run()


def compare_with_reference(
    results_excel: str,
    reference_csv: str,
    approach: str = "GHB",
) -> None:
    """
    Compara las cabezas en piezómetros del Excel de resultados del módulo
    contra el CSV de referencia del modelo original (Model1.ob_gw_out_head.csv).
    Imprime tabla: piezómetro | módulo | referencia | diferencia.
    """
    # Leer resultados del módulo desde Excel
    df_piezo = pd.read_excel(results_excel, sheet_name="Piezometros")
    col_modulo = f"cabeza_{approach}_m"
    if col_modulo not in df_piezo.columns:
        # Si solo hay un enfoque, tomar la primera columna de cabeza disponible
        col_modulo = [c for c in df_piezo.columns if c.startswith("cabeza_")][0]

    # Leer referencia CSV (formato: time, PIEZO1, PIEZO2, ...)
    df_ref = pd.read_csv(reference_csv)
    last = df_ref.iloc[-1]
    ref_cols = {c.strip().upper(): float(last[c]) for c in df_ref.columns[1:]}

    print(f"\n{'='*60}")
    print(f"Comparación módulo WaterKu vs referencia original [{approach}]")
    print(f"{'='*60}")
    print(f"{'Piezometro':<14} {'Modulo (m)':<14} {'Referencia (m)':<16} {'Dif (m)':<10}")
    print("-" * 60)

    for _, row in df_piezo.iterrows():
        name = str(row["piezometro"]).strip()
        val_modulo = float(row[col_modulo])
        val_ref = ref_cols.get(name.upper(), float("nan"))
        delta = val_modulo - val_ref
        print(f"{name:<14} {val_modulo:<14.4f} {val_ref:<16.4f} {delta:+.4f}")

    print("="*60)
    print("Objetivo: |Dif| < 0.01 m para replicacion exacta")


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/config.modflow_laguna.yaml"
    process_modflow_laguna(cfg)
