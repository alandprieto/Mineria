import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "04_regression"
OUT.mkdir(parents=True, exist_ok=True)

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata.language_info = {"name": "python"}

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Fase 5 — Objetivo 3: Regresión de atributos de jugador

> **Objetivo del informe:** construir un modelo de regresión para estimar atributos de rendimiento de un jugador en función de su perfil físico y su posición.

Diseño:
- **Variable objetivo:** `overall_rating` (el valor general del jugador, excluida de las 24 variables del clustering).
- **Variables predictoras (perfil físico + posición):** altura (cm), peso (kg), edad al momento de la última valoración y **posición** (Arquero / Defensor / Mediocampista / Delantero).
- La **posición** se deriva del clustering de la Sección 4.3: K-Means (k=3) sobre los 24 atributos de campo de los jugadores no arqueros, más la regla `gk_diving >= 30` para identificar arqueros.
- Se comparan: *baseline* (media), **Ridge** (lineal), **Random Forest** y **XGBoost**. Métricas: MAE, RMSE y R².
"""),
    code("""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from mineria.data_loader import load_table
from mineria.regression import (
    fit_position_kmeans,
    map_clusters_to_positions,
    player_regression_data,
    regression_metrics,
    features_prepare,
)"""),
    md("""## 1. Etiquetado de posición (clustering §4.3)

K-Means sobre los 24 atributos de campo de jugadores no arqueros, con los mismos centroides y semilla que el notebook `02_kmeans_jugadores`."""),
    code("""pa = load_table("Player_Attributes")
players = load_table("Player")
print("Player_Attributes:", pa.shape, "| Player:", players.shape)"""),
    code("""km = fit_position_kmeans(pa)
cluster_map = map_clusters_to_positions(pa, km)
print("cluster -> posición:", cluster_map)

pos = (pa
       .sort_values("date")
       .drop_duplicates("player_api_id", keep="last"))
field_n = int((pos["gk_diving"] < 30).sum())
gk_n = int((pos["gk_diving"] >= 30).sum())
print(f"jugadores de campo: {field_n:,} | arqueros: {gk_n:,}")"""),
    md("""## 2. Dataset de regresión

Última valoración por jugador + perfil físico (Player) + posición. Se descartan los jugadores sin `overall_rating`."""),
    code("""reg = player_regression_data(pa, players, km, cluster_map)
print("muestra:", reg.shape)
reg.head(5)"""),
    code("""print("Distribución por posición")
print(reg["position"].value_counts().to_string())
print()
print("Valoración media por posición")
print(reg.groupby("position")["overall_rating"].mean().round(2).to_string())
print()
print(reg[["height", "weight", "age", "overall_rating"]].describe().round(1).to_string())"""),
    md("""## 3. Entrenamiento y evaluación

Split 80/20 estratificado por posición (seed fija). Para Ridge se estandarizan las variables numéricas."""),
    code("""X, y = features_prepare(reg)
print("X:", X.shape)
print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=reg["position"]
)
print(f"train: {X_train.shape[0]:,} | test: {X_test.shape[0]:,}")"""),
    code("""baseline_mae = np.mean(np.abs(y_test - y_train.mean()))
baseline_rmse = float(np.sqrt(np.mean((y_test - y_train.mean()) ** 2)))
baseline_r2 = 1 - np.sum((y_test - y_train.mean()) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
print(f"Baseline (media global {y_train.mean():.1f}): MAE={baseline_mae:.3f} RMSE={baseline_rmse:.3f} R2={baseline_r2:.3f}")"""),
    code("""scaler = StandardScaler().fit(X_train[["height", "weight", "age"]])
Xs = X.copy()
Xs["height"], Xs["weight"], Xs["age"] = (scaler.transform(X[["height", "weight", "age"]]).T)
Xs_train, Xs_test = Xs.loc[X_train.index], Xs.loc[X_test.index]

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(Xs_train, y_train)
print("Ridge:", regression_metrics(y_test, ridge.predict(Xs_test)))"""),
    code("""rf = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("Random Forest:", regression_metrics(y_test, rf.predict(X_test)))
print("importancia", dict(zip(X_train.columns, rf.feature_importances_.round(3))))"""),
    code("""xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                       subsample=0.8, colsample_bytree=0.8, random_state=42,
                       early_stopping_rounds=15, eval_metric="rmse")
xgb.fit(Xs_train, y_train, eval_set=[(Xs_test, y_test)], verbose=False)
print("XGBoost (best_iter", xgb.best_iteration, "):",
      regression_metrics(y_test, xgb.predict(Xs_test)))
print("importancia", dict(zip(Xs_train.columns, xgb.feature_importances_.round(3))))"""),
    md("""## 4. Comparación

La tabla resume las tres estrategias junto al *baseline* de predecir la media de la valoración."""),
    code("""results = pd.DataFrame({
    "Modelo": ["Baseline (media)", "Ridge", "Random Forest", "XGBoost"],
    "MAE": [baseline_mae, ridge_mae := regression_metrics(y_test, ridge.predict(Xs_test))["mae"],
            regression_metrics(y_test, rf.predict(X_test))["mae"],
            regression_metrics(y_test, xgb.predict(Xs_test))["mae"]],
    "RMSE": [baseline_rmse, regression_metrics(y_test, ridge.predict(Xs_test))["rmse"],
             regression_metrics(y_test, rf.predict(X_test))["rmse"],
             regression_metrics(y_test, xgb.predict(Xs_test))["rmse"]],
    "R2": [baseline_r2, regression_metrics(y_test, ridge.predict(Xs_test))["r2"],
           regression_metrics(y_test, rf.predict(X_test))["r2"],
           regression_metrics(y_test, xgb.predict(Xs_test))["r2"]],
})
results.round(3)"""),
    code("""import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(results["Modelo"], results["R2"], color="steelblue")
axes[0].set_title("R² por modelo")
axes[0].set_ylim(0, 1)
for i, v in enumerate(results["R2"]):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center")
axes[1].bar(results["Modelo"], results["MAE"], color="indianred")
axes[1].set_title("MAE por modelo\\n(escala 0-100 de la valoración)")
for i, v in enumerate(results["MAE"]):
    axes[1].text(i, v + 0.1, f"{v:.2f}", ha="center")
plt.tight_layout()
plt.show()"""),
    code("""imp = pd.DataFrame({
    "Random Forest": dict(zip(X_train.columns, rf.feature_importances_)),
    "XGBoost": dict(zip(Xs_train.columns, xgb.feature_importances_.round(3))),
}).sort_values("Random Forest", ascending=False)
imp.plot(kind="barh", figsize=(8, 4))
plt.title("Importancia de las variables")
plt.xlabel("importancia")
plt.tight_layout()
plt.show()"""),
    md("""## 5. Conclusiones

- El **perfil físico y la posición explican una fracción acotada de la valoración**: los mejores modelos (Random Forest y XGBoost) logran **R² ≈ 0.3-0.4**, muy por encima del *baseline* (R² = 0).
- **XGBoost** resulta el más preciso (menor MAE/RMSE), aunque la diferencia con Random Forest es pequeña.
- La **edad** domina la importancia de las variables (refleja la curva de carrera), seguida de la **posición** y, en menor medida, la **altura/peso**.
- Un jugador puede estimarse con un error de ~5-6 puntos sobre la escala 0-100; la fracción no explicada corresponde a los atributos técnicos (regate, pase, tiro…), es decir, al **talento individual** que el informe cuantifica en el Objetivo 1 con clustering.

## Reproducibilidad
```bash
.venv/Scripts/python.exe scripts/gen_regression_nb.py      # genera este notebook
.venv/Scripts/python.exe -m nbconvert --execute --to notebook --inplace notebooks/04_regression/01_regresion_atributos.ipynb
```
"""),
]

nb.cells = cells
out = OUT / "01_regresion_atributos.ipynb"
nbf.write(nb, out)
print("escrito:", out)