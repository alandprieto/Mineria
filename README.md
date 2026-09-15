# Minería de Datos — TP Integrador

Predicción de resultados de fútbol europeo y análisis de perfiles de jugadores/equipos a partir de la [European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer) (Kaggle), correspondiente al Trabajo Práctico Integrador bajo el informe `informe/TPIG2.md`.

## Objetivos (según el informe)

1. **Clustering**: identificar arquetipos de jugadores y estilos tácticos de equipos (K-Medias).
2. **Clasificación**: predecir el resultado de un partido (victoria local, empate, victoria visitante).
3. **Regresión**: estimar atributos de rendimiento de un jugador según su perfil físico y posición.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .               # instala el paquete src/mineria/
```

Descargar `database.sqlite` desde Kaggle (European Soccer Database) y ubicarlo en:

```
data/database.sqlite
```

> El archivo queda fuera de git (`.gitignore`). Sin él los notebooks no se pueden ejecutar.

Para los notebooks:

```
jupyter notebook notebooks/
```

## Reproducción

Los notebooks canónicos ya vienen ejecutados con sus salidas. Para regenerarlos desde cero:

```bash
# 1. Pipeline de clasificación (5 recetas, tabla 7 del informe)
.venv/Scripts/python.exe scripts/experiments.py

# 2. Notebooks canónicos (re-ejecución in-place con salidas frescas)
for nb in notebooks/02_clustering/01_kmeans_equipos.ipynb \
          notebooks/02_clustering/02_kmeans_jugadores.ipynb \
          notebooks/03_classification/00_reporte_xgboost.ipynb \
          notebooks/03_classification/01_mitigacion_tabnet.ipynb; do
  .venv/Scripts/python.exe -m nbconvert --execute --to notebook --inplace "$nb"
done

.venv/Scripts/python.exe -m nbconvert --execute --to notebook --inplace notebooks/03_classification/Pred-Racha.ipynb
.venv/Scripts/python.exe scripts/gen_regression_nb.py
.venv/Scripts/python.exe -m nbconvert --execute --to notebook --inplace notebooks/04_regression/01_regresion_atributos.ipynb
```

Notas de ejecución:

- `experiments.py` y los notebooks canónicos resuelven las rutas de datos/modelos a través de `src/mineria/config.py` (independiente del directorio actual).
- `Pred-Racha.ipynb` usa rutas relativas (`../../data/...`) y debe ejecutarse desde su propio directorio (`notebooks/03_classification/`), o directamente en Jupyter.
- `Pred-Racha.ipynb` entrena su propio modelo y lo guarda en `models/modelo_pred_racha_analisis.joblib` (sin tocar el artefacto canónico consumido por `src/mineria/inference.py`).

## Estructura

```
├── data/                  database.sqlite (descargar manualmente, no versionado)
├── informe/               Informe final (TPIG2.md / TPIG2.pdf)
├── models/                Modelos serializados (*.joblib, no versionados)
├── notebooks/
│   ├── 02_clustering/     Clustering jugadores y equipos — informe §4.2 / §4.3
│   ├── 03_classification/ Predicción H/D/A con XGBoost, cuotas, rachas y TabNet
│   └── 04_regression/     Regresión de atributos — objetivo 3
├── scripts/               Validación terminal (experiments.py)
└── src/mineria/           Código reutilizable (rutas, carga, features, modelos, clustering, plots)
```

## Estado del proyecto vs informe

| Componente | Estado |
|---|---|
| EDA (tablas y cuotas de apuestas) | Implementado (`notebooks/01_eda`, `03_classification/Pred-*`) |
| Clustering equipos (§4.2, K=3, 489 snapshots) | Implementado — DB≈2.26, tamaños 137/186/166, representativos Brugge/Charleroi/Sociedad |
| Clustering jugadores (§4.3, K=3, 20k muestra) | Implementado — DB≈1.56, perfiles Defensor/Medio/Delantero, representativos McGivern/Dessena/Nijland |
| Clasificación XGBoost (PA/TA/odds/racha, ~52% test) | Implementado (`00_reporte_xgboost.ipynb`) |
| Regularización + early stopping (§5.3) | Implementado — train 56.89%, test 52.41%, gap 4.48pp (`01_mitigacion_tabnet.ipynb`) |
| Búsqueda con Optuna (§5.3, 50 trials) | Implementado — mejor mlogloss 0.966, test 51.54% |
| Red neuronal TabNet (§5.4, 51.12% reportado) | Implementado — test 50.99%, celda de `Pred-Racha` corregida |
| Regresión atributos de jugador (objetivo 3) | Implementado — XGBoost MAE 4.42, RMSE 5.45, R² 0.239 (`01_regresion_atributos.ipynb`) |

### Resultados de clustering (coincidencia con el informe)

| Metrica | Equipos | Jugadores |
|---|---|---|
| N total | 489 | 20,000 |
| DB @ K=3 | 2.26 (informe: 2.51→2.10 rango K=2..6) | 1.56 (informe: 1.54) |
| Sizes | 137 / 186 / 166 | 7831 / 5643 / 6526 |
| Representativos | Club Brugge KV / Sporting Charleroi / Real Sociedad | Stefan Nijland / Ryan McGivern / Daniele Dessena |
| Perfiles | Conservador / Contraataque / Ofensivo | Delantero / Defensor / Mediocampista |

## Fidelidad al informe

Todos los resultados de este repo son **realmente computados** sobre `data/database.sqlite` con el código de `src/mineria/`. Las diferencias observadas frente a `informe/TPIG2.md` se deben a versiones de librerías y a la variación propia de los procedimientos estocásticos, y **no** a ajuste de resultados:

- **Clustering jugadores (§4.3)**: DB 1.562 vs 1.54 del informe; tamaños de cluster 7831/5643/6526 vs 7694/5813/6493. El conteo de elegibles (166.432) y las medias de perfil coinciden.
- **TabNet (§5.4)**: exactitud 50.99% vs 51.12%; el patrón de predicción (recall alto en victoria local, casi nulo en empate) se reproduce.
- **Optuna (§5.3)**: nuestro mejor estudio (mlogloss valid 0.966, test 51.54%) difiere del del informe por la búsqueda estocástica; ambos logran el mismo rango de precisión.
- **Regresión (objetivo 3)**: es un diseño propio alineado al enunciado (el informe no detalla la sección); R² 0.239 con XGBoost sobre perfil físico + posición.

## Datos de referencia del informe

- Umbral de rentabilidad (breakeven): `1 / 1.85 ≈ 54.05%`.
- Hiperparámetros base XGBoost: `multi:softprob`, `num_class=3`, `mlogloss`, 200 estimadores, `max_depth=4`, `learning_rate=0.05`, `tree_method=hist`, `random_state=42`.
- Casas de apuestas en orden por cobertura: B365 (13.04%) < BW < WH < VC < LB < IW < SJ (34.19%) < GB < BS < PS (57.01%).