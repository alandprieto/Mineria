import pandas as pd


class MedianImputer:
    def __init__(self):
        self.medians_ = None

    def fit(self, X: pd.DataFrame) -> "MedianImputer":
        self.medians_ = X.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.fillna(self.medians_)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)