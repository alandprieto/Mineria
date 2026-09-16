import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

from mineria.config import DB_PATH

sns.set_theme(style="whitegrid")
PAL = sns.color_palette("Set2")
OUT = "graficos"


def fig_clasificacion():
    recipes = ["Team", "Player", "Player+Team", "Cuotas B365", "Combinado"]
    train = [52.95, 60.86, 60.87, 52.59, 59.77]
    test = [50.54, 52.00, 51.92, 52.19, 52.42]
    breakeven = 100 / 1.85
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = np.arange(len(recipes))
    w = 0.36
    ax.bar(x - w / 2, train, w, label="Entrenamiento", color=PAL[0])
    ax.bar(x + w / 2, test, w, label="Test", color=PAL[1])
    ax.axhline(breakeven, ls="--", color="crimson", lw=1.5)
    ax.text(len(recipes) - 0.35, breakeven + 0.4, f"Breakeven {breakeven:.2f}%", color="crimson", fontsize=10, ha="right")
    ax.set_xticks(x, recipes)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("XGBoost — accuracy por receta de features", fontsize=13)
    ax.set_ylim(45, 66)
    for i in range(len(recipes)):
        ax.text(i + w / 2, test[i] + 0.5, f"{test[i]:.2f}", ha="center", fontsize=9)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/clasificacion_modelos.png", dpi=150)
    plt.close(fig)


def fig_mitigacion():
    models = ["Original", "Regularizado", "Optuna", "TabNet"]
    train = [60.52, 56.89, 66.62, None]
    test = [52.48, 52.41, 51.54, 50.99]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(models))
    w = 0.36
    tw = [t for t in train if t is not None]
    ax.bar(x[: len(tw)] - w / 2, tw, w, label="Entrenamiento", color=PAL[0])
    ax.bar(x + w / 2, test, w, label="Test", color=PAL[1])
    for i in range(len(models)):
        ax.text(i, test[i] + 0.6, f"{test[i]:.2f}%", ha="center", fontsize=10)
    for i in range(len(tw)):
        ax.text(i - 0.2, tw[i] + 0.6, f"{tw[i]:.2f}%", ha="center", fontsize=9)
    ax.set_xticks(x, models)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Mitigación del sobreajuste — train vs test", fontsize=13)
    ax.set_ylim(45, 72)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/mitigacion_sobreajuste.png", dpi=150)
    plt.close(fig)


def fig_regresion():
    models = ["Baseline", "Ridge", "Random Forest", "XGBoost"]
    mae = [5.034, 4.558, 4.537, 4.417]
    rmse = [6.264, 5.656, 5.629, 5.454]
    r2 = [-0.004, 0.182, 0.189, 0.239]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(models))
    w = 0.36
    ax.bar(x - w / 2, mae, w, label="MAE", color=PAL[0])
    ax.bar(x + w / 2, rmse, w, label="RMSE", color=PAL[2])
    for i in range(len(models)):
        ax.text(i, rmse[i] + 0.12, r"$R^2$" + f" {r2[i]:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, models)
    ax.set_ylabel("Error (puntos sobre 100)")
    ax.set_title("Regresión de overall_rating — perfil físico + posición", fontsize=13)
    ax.set_ylim(0, 7.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/regresion_atributos.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_clasificacion()
    fig_mitigacion()
    fig_regresion()
    print("Gráficos guardados en", OUT)
    print("Nota: las figuras fig4_*.png y las de graficos_exportados/ son generadas externamente.")
