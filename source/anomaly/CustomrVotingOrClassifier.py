from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class CustomrVotingOrClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators=None, voting=None):
        self.voting = voting
        self.estimators = estimators
        self.predictions = list()

    def fit(self, X, y=None):
        for classifier in self.estimators:
            classifier.fit(X, y)

    def predict(self, X, y=None):
        for classifier in self.estimators:
            self.predictions.append(classifier.predict(X))

        # return np.mean(self.predictions, axis=0).round(0)
        return np.max(self.predictions, axis=0)
