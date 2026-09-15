import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mineria.pipeline import run_experiment


def main():
    for recipe in ["ta", "pa", "odds", "pa_ta", "combined"]:
        result = run_experiment(recipe)
        print(
            f"{recipe:10s} acc_train={result['train_accuracy']:.4f} "
            f"acc_test={result['accuracy']:.4f} breakeven={result['breakeven']:.4f}"
        )


if __name__ == "__main__":
    main()