from xgboost import XGBClassifier

from ..config import SEED


def default_params(**overrides):
    params = dict(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        tree_method="hist",
        random_state=SEED,
    )
    params.update(overrides)
    return params


def regularized_params(**overrides):
    params = default_params()
    params.update(
        dict(
            reg_lambda=5.0,
            reg_alpha=1.5,
            gamma=0.5,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    )
    params.update(overrides)
    return params


def train_xgb(
    X_train,
    y_train,
    params=None,
    eval_set=None,
    early_stopping_rounds=None,
    verbose=False,
    **overrides,
):
    if params is None:
        params = default_params()
    params = dict(params)
    params.update(overrides)
    if early_stopping_rounds is not None:
        params["early_stopping_rounds"] = early_stopping_rounds
    model = XGBClassifier(**params)
    fit_kwargs = {}
    if eval_set is not None:
        fit_kwargs["eval_set"] = eval_set
    model.fit(X_train, y_train, verbose=verbose, **fit_kwargs)
    return model