"""
Utilidades compartidas para scripts de hidrología
"""
import os
import yaml


def load_cuenca_config(cuenca_id):
    """
    Cargar configuración específica de la cuenca.
    Soporta tanto el nuevo formato (config.basins.yaml) como el legacy (inline en config.yaml).

    Args:
        cuenca_id: ID de la cuenca a cargar

    Returns:
        tuple: (cuenca_config, main_config)
            - cuenca_config: Diccionario con configuración de la cuenca específica
            - main_config: Diccionario con toda la configuración principal

    Raises:
        ValueError: Si la cuenca no existe en la configuración
    """
    # Load main config
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        main_config = yaml.safe_load(f)

    # Check if basins_config reference exists (new format)
    if 'basins_config' in main_config:
        basins_config_path = main_config['basins_config']
        with open(basins_config_path, "r", encoding="utf-8") as f:
            basins_config = yaml.safe_load(f)

        # Merge basins into main config
        cuencas = basins_config.get('cuencas', {})

        # Also get global usar_parametros_excel if present
        if 'usar_parametros_excel' in basins_config:
            main_config['usar_parametros_excel'] = basins_config['usar_parametros_excel']

    # Check for inline cuencas (current format, might still be present)
    elif 'cuencas' in main_config:
        cuencas = main_config['cuencas']

    # Check for deprecated inline format
    elif 'cuencas_inline_deprecated' in main_config:
        cuencas = main_config['cuencas_inline_deprecated']

    else:
        raise ValueError("No se encontró configuración de cuencas (busque 'basins_config' o 'cuencas' en config.yaml)")

    # Get specific basin config
    if cuenca_id not in cuencas:
        raise ValueError(f"Cuenca '{cuenca_id}' no encontrada en configuración. Cuencas disponibles: {list(cuencas.keys())}")

    cuenca_config = cuencas[cuenca_id]

    # Inherit global usar_parametros_excel if not specified per-basin
    if 'usar_parametros_excel' not in cuenca_config and 'usar_parametros_excel' in main_config:
        cuenca_config['usar_parametros_excel'] = main_config['usar_parametros_excel']

    return cuenca_config, main_config


def should_use_excel(parametros_path, cuenca_config):
    """
    Determina si se deben usar parámetros desde Excel según la configuración y el path.

    Args:
        parametros_path: Ruta al archivo de parámetros
        cuenca_config: Diccionario de configuración de la cuenca

    Returns:
        bool: True si se deben usar parámetros de Excel, False en caso contrario
    """
    usar_excel = cuenca_config.get('usar_parametros_excel', True)

    # Si el path contiene config.yaml, no usar Excel
    if 'config.yaml' in parametros_path:
        return False

    # Si el archivo no existe, no usar Excel
    if not os.path.exists(parametros_path):
        return False

    return usar_excel
