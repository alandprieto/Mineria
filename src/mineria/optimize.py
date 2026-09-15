import optuna
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from .config import SEED


def make_objective(X_train, y_train, X_valid, y_valid):
    def objective(trial):
        params = dict(
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=SEED,
            reg_lambda=5.0,
            reg_alpha=1.5,
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0.01, 1.0, log=True),
            early_stopping_rounds=10,
        )
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        y_pred = model.predict_proba(X_valid)
        return log_loss(y_valid, y_pred, labels=[0, 1, 2])

    return objective


def run_study(
    X_train,
    y_train,
    X_valid,
    y_valid,
    n_trials=50,
    seed=SEED,
    show_progress_bar=False,
):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        make_objective(X_train, y_train, X_valid, y_valid),
        n_trials=n_trials,
        show_progress_bar=show_progress_bar,
    )
    return study


def best_model_from_study(study, X_train, y_train):
    params = dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        reg_lambda=5.0,
        reg_alpha=1.5,
        **study.best_params,
    )
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model