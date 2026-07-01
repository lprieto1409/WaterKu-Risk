# data/valle_real/

Datos de entrada para el proyecto "Evaluación Hidrológica e Hidrogeológica de la Laguna Colombina
Sur — Habilitación Urbana Valle Real" (ARKEL S.A.C, Opción 3). Ninguno de estos datos existe todavía
en el repo; se cargan aquí conforme se reciben.

- `dem/` — DEM de la cuenca/microcuencas aportantes a la laguna (GeoTIFF). Resolución a evaluar
  (ASTER GDEM 30m probablemente insuficiente para una cuenca pequeña; preferir cartografía 1:25k IGN
  o topografía de ARKEL).
- `senamhi/` — series históricas de las 4 estaciones del valle del Mantaro (Viques, Huayao, Santa Ana,
  Ingenio), adquiridas vía TUPA SENAMHI.
- `pisco_era5/` — grillas de reanálisis PISCO (precipitación, SENAMHI) y ERA5 (ECMWF, variables
  meteorológicas) recortadas al área de la cuenca.
- `sondeos/` — logs de las 2 perforaciones washboring de 15 m con SPT (entregados por ARKEL).
- `geofisica/` — perfiles MASW-2D y de tomografía de resistividad eléctrica (ERT) ya procesados
  (entregados por ARKEL o su subcontratista de geofísica). WaterKu-Risk solo importa estos resultados,
  no reimplementa inversión geofísica.
- `laboratorio/` — resultados de ensayos Lefranc/Porchet, permeámetro, y análisis químicos (sales
  solubles, sulfatos, cloruros), entregados por ARKEL.
