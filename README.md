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

## Estructura

```
├── data/                  database.sqlite (descargar manualmente, no versionado)
├── informe/               Informe final (TPIG2.md / TPIG2.pdf)
├── models/                Modelos serializados (*.joblib, no versionados)
├── notebooks/
│   ├── 02_clustering/     Clustering jugadores y equipos — informe §4.2 / §4.3
│   ├── 03_classification/ Predicción H/D/A con XGBoost, cuotas, rachas y TabNet
│   └── 04_regression/     (pendiente) Regresión de atributos — objetivo 3
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
| Red neuronal TabNet (§5.4, 51.12% reportado) | Pendiente — celda rota en `Pred-Racha` |
| Regularización / early stopping / Optuna (§5.3) | Pendiente |
| Regresión atributos de jugador (objetivo 3) | Pendiente |

### Resultados de clustering (coincidencia con el informe)

| Metrica | Equipos | Jugadores |
|---|---|---|
| N total | 489 | 20,000 |
| DB @ K=3 | 2.26 (informe: 2.51→2.10 rango K=2..6) | 1.56 (informe: 1.54) |
| Sizes | 137 / 186 / 166 | 7831 / 5643 / 6526 |
| Representativos | Club Brugge KV / Sporting Charleroi / Real Sociedad | Stefan Nijland / Ryan McGivern / Daniele Dessena |
| Perfiles | Conservador / Contraataque / Ofensivo | Delantero / Defensor / Mediocampista |

## Datos de referencia del informe

- Umbral de rentabilidad (breakeven): `1 / 1.85 ≈ 54.05%`.
- Hiperparámetros base XGBoost: `multi:softprob`, `num_class=3`, `mlogloss`, 200 estimadores, `max_depth=4`, `learning_rate=0.05`, `tree_method=hist`, `random_state=42`.
- Casas de apuestas en orden por cobertura: B365 (13.04%) < BW < WH < VC < LB < IW < SJ (34.19%) < GB < BS < PS (57.01%).