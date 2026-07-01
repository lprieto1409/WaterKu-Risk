# =============================================================================
# HIDROGEOLOGÍA — Transporte de contaminantes (ADE / MT3DMS)
# =============================================================================

_mt3dms_out = config.get("output_base", "results") + "/hidrogeologia/mt3dms"
_transport_cfg = "config/config.transport.yaml"


rule captura_contaminantes:
    input:
        emitters=config.get(
            "emitters_geojson",
            "data/child_health/real/emitters.geojson",
        ),
        settlements=config.get(
            "settlements_geojson",
            "data/child_health/real/settlements.geojson",
        ),
        config=_transport_cfg,
    output:
        breakthrough=f"{_mt3dms_out}/breakthrough_curves.png",
        mass_balance=f"{_mt3dms_out}/mass_balance.png",
        summary=f"{_mt3dms_out}/concentration_summary.xlsx",
    script:
        "../scripts/transport/transport_runner.py"


rule benchmark_1d:
    output:
        f"{_mt3dms_out}/benchmark_1d.png",
    script:
        "../scripts/transport/benchmarks.py"


rule benchmark_3d:
    output:
        f"{_mt3dms_out}/benchmark_3d.png",
    script:
        "../scripts/transport/benchmarks.py"


rule modflow_laguna:
    input:
        excel=config.get("modflow_laguna_excel", "data/modflow_laguna/modflow_laguna_data.xlsx"),
        modflow_config="config/config.modflow_laguna.yaml",
    output:
        excel=f"{_mt3dms_out}/modflow_laguna/modflow_laguna_resultados.xlsx",
        png=f"{_mt3dms_out}/modflow_laguna/cabezas_lak6.png",
    script:
        "../scripts/modflow/run_modflow_laguna.py"


rule hidrogeologia:
    input:
        rules.captura_contaminantes.output.summary,
        rules.captura_contaminantes.output.breakthrough,
        rules.captura_contaminantes.output.mass_balance,
        rules.benchmark_1d.output,
        rules.benchmark_3d.output,
        rules.modflow_laguna.output.excel,
