import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, ClassifierMixin


class CustomIsolationForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators, contamination):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=27,
            verbose=0)

    def fit(self, X, y=None):
        self.isolation_forest.fit(X, y)

    def predict(self, X, y=None):
        return np.where(
            self.isolation_forest.predict(X) == 1, 0, 1)

    def predict_proba(self, X, y=None):
        proba = self.isolation_forest.decision_function(X)
        return np.asarray([1 - proba, proba]).T
