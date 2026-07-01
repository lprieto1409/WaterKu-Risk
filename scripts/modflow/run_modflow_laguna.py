"""
Wrapper CLI / Snakemake para el módulo modflow_laguna (WaterKu-Risk).

Uso:
    python scripts/modflow/run_modflow_laguna.py --config config/config.modflow_laguna.yaml
    python waterku_risk.py modflow-laguna
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.modflow.modflow_laguna import process_modflow_laguna  # noqa: E402


def _get_snakemake_config():
    try:
        config_path = snakemake.input.modflow_config  # noqa: F821
        output_dir = str(Path(snakemake.output.excel).parent)  # noqa: F821
        return config_path, output_dir
    except NameError:
        return None, None


def main():
    config_path, output_dir = _get_snakemake_config()

    if config_path is None:
        import argparse
        parser = argparse.ArgumentParser(
            description="WaterKu-Risk — Módulo MODFLOW 6 Lagoon-Aquifer"
        )
        parser.add_argument(
            "--config", "-c",
            default=str(_ROOT / "config" / "config.modflow_laguna.yaml"),
        )
        parser.add_argument("--output-dir", "-o", default=None)
        args = parser.parse_args()
        config_path = args.config
        output_dir = args.output_dir

    process_modflow_laguna(config_path, output_dir)


if __name__ == "__main__":
    main()
