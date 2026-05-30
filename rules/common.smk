import yaml
import os

# Directorio base de salida
output_base = config.get("output_base", "results")
os.makedirs(f"{output_base}/hidrogeologia/mt3dms", exist_ok=True)

# Cargar configuración de transporte
project_root = workflow.basedir

with open(os.path.join(project_root, "config/config.transport.yaml")) as f:
    transport_config = yaml.safe_load(f)

with open(os.path.join(project_root, "config/config.modflow.yaml")) as f:
    modflow_config = yaml.safe_load(f)
