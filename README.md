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
│   ├── 00_tutorials/      Guías P0–P6 (prácticas)
│   ├── 01_eda/            Exploración: Country, League, Team, Match, Player/Team Attributes
│   ├── 02_clustering/     (pendiente) Clustering jugadores y equipos — informe §4.2 / §4.3
│   ├── 03_classification/ Predicción H/D/A con XGBoost, cuotas, rachas y TabNet
│   └── 04_regression/     (pendiente) Regresión de atributos — objetivo 3
└── src/mineria/           Código reutilizable (rutas, carga de datos, features, modelos)
```

## Estado del proyecto vs informe

| Componente | Estado |
|---|---|
| EDA (tablas y cuotas de apuestas) | Implementado (`notebooks/01_eda`, `03_classification/Pred-*`) |
| Clasificación XGBoost (PA/TA/odds/racha, ~52% test) | Implementado |
| Clustering jugadores y equipos (§4.2 / §4.3) | **Pendiente** |
| Red neuronal TabNet (§5.4, 51.12% reportado) | **Pendiente — celda rota en `Pred-Racha`** |
| Regularización / early stopping / Optuna (§5.3) | **Pendiente** |
| Regresión atributos de jugador (objetivo 3) | **Pendiente** |

## Datos de referencia del informe

- Umbral de rentabilidad (breakeven): `1 / 1.85 ≈ 54.05%`.
- Hiperparámetros base XGBoost: `multi:softprob`, `num_class=3`, `mlogloss`, 200 estimadores, `max_depth=4`, `learning_rate=0.05`, `tree_method=hist`, `random_state=42`.
- Casas de apuestas en orden por cobertura: B365 (13.04%) < BW < WH < VC < LB < IW < SJ (34.19%) < GB < BS < PS (57.01%).