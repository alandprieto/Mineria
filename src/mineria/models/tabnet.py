from pytorch_tabnet.tab_model import TabNetClassifier

from ..config import SEED


def train_tabnet(
    X_train,
    y_train,
    X_valid=None,
    y_valid=None,
    n_d=32,
    n_a=32,
    n_steps=5,
    max_epochs=200,
    patience=30,
    batch_size=1024,
    virtual_batch_size=128,
    verbose=0,
    **overrides,
):
    """Entrena TabNetClassifier (los datos se pasan como .values, requisito de la API)."""
    model = TabNetClassifier(
        n_d=n_d,
        n_a=n_a,
        n_steps=n_steps,
        seed=SEED,
        verbose=verbose,
        **overrides,
    )
    eval_set = [(X_valid.values, y_valid.values)] if X_valid is not None else None
    model.fit(
        X_train.values,
        y_train.values,
        eval_set=eval_set,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size,
    )
    return model


def predict_tabnet(model, X):
    return model.predict(X.values)


def predict_proba_tabnet(model, X):
    return model.predict_proba(X.values)


def tabnet_scores(model, X_test, y_test):
    from ..evaluate import evaluate

    return evaluate(model, X_test.values, y_test)