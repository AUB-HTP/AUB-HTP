import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import validate_data, check_is_fitted

from .covariance import HeavyTailedCovariance
from ._shape import MethodLiteral


class HeavyTailedPCA(TransformerMixin, BaseEstimator):
    def __init__(self):
        raise NotImplementedError("HeavyTailedPCA is not implemented yet.")


    def fit(self, X, y=None):
        raise NotImplementedError("HeavyTailedPCA is not implemented yet.")


    def transform(self, X):
        raise NotImplementedError("HeavyTailedPCA is not implemented yet.")

    def score(self, X, y=None):
        raise NotImplementedError("HeavyTailedPCA is not implemented yet.")