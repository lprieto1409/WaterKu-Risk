# =============================================================================
# VALLE REAL — Laguna Colombina Sur (Propuesta Arkel, Opción 3)
# =============================================================================
# Paralelo a rules/hidrogeologia.smk (que no se modifica). Cada regla sigue el
# mismo patrón: input.config -> scripts/<dominio>/<dominio>_runner.py -> output
# en results/<dominio>_vallereal/.

_vr_output_base = config.get("output_base", "results")


rule hidrologia_vallereal:
    input:
        config="config/config.hidrologia.yaml",
    output:
        directory(f"{_vr_output_base}/hidrologia_vallereal"),
    script:
        "../scripts/hidrologia/hidrologia_runner.py"


rule geotecnia_vallereal:
    input:
        config="config/config.geotecnia.yaml",
    output:
        f"{_vr_output_base}/geotecnia_vallereal/modelo_conceptual.json",
    script:
        "../scripts/geotecnia/geotecnia_runner.py"


rule modflow_vallereal:
    input:
        config="config/config.modflow_vallereal.yaml",
        modelo_conceptual=rules.geotecnia_vallereal.output,
    output:
        f"{_vr_output_base}/modflow_vallereal/resumen_modflow_vallereal.json",
    script:
        "../scripts/modflow/model_builder.py"


rule balance_vallereal:
    input:
        config="config/config.balance.yaml",
        modflow_summary=rules.modflow_vallereal.output,
        hidrologia=rules.hidrologia_vallereal.output,
    output:
        f"{_vr_output_base}/balance_vallereal/balance_hidrico_laguna_colombina_sur.xlsx",
    script:
        "../scripts/balance/balance_runner.py"


rule vallereal_all:
    input:
        rules.hidrologia_vallereal.output,
        rules.geotecnia_vallereal.output,
        rules.modflow_vallereal.output,
        rules.balance_vallereal.output,
