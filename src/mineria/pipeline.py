from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .config import DB_PATH, SEED
from .dataset import build_dataset
from .evaluate import evaluate
from .impute import MedianImputer
from .models.xgb import train_xgb


def run_experiment(recipe, db_path=DB_PATH, test_size=0.2, seed=SEED, **xgb_overrides):
    features, feature_columns = build_dataset(recipe, db_path=db_path)
    X, y = features[feature_columns], features["FTR"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    imputer = MedianImputer()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = train_xgb(X_train, y_train, **xgb_overrides)

    return {
        "recipe": recipe,
        "model": model,
        "imputer": imputer,
        "feature_columns": feature_columns,
        "X_test": X_test,
        "train_accuracy": round(accuracy_score(y_train, model.predict(X_train)), 4),
        **evaluate(model, X_test, y_test),
    }