from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config import BREAKEVEN

TARGET_NAMES = ["Home Win", "Draw", "Away Win"]


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "breakeven": round(BREAKEVEN, 4),
        "report": classification_report(
            y_test, y_pred, target_names=TARGET_NAMES, output_dict=True
        ),
        "confusion": confusion_matrix(y_test, y_pred),
        "y_test": y_test,
        "y_pred": y_pred,
    }