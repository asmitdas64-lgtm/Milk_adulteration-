"""Custom Scikit-Learn transformers for physicochemical preprocessing."""

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features to [Q1 - factor*IQR, Q3 + factor*IQR] learned on fit data.

    Prevents extreme anomalous values from distorting distance-based algorithms
    while preserving the samples and their class labels (decision D-05).
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_: Optional[np.ndarray] = None
        self.upper_: Optional[np.ndarray] = None

    def fit(self, X, y=None):
        """Compute IQR bounds per column on training data."""
        X_arr = np.asarray(X, dtype=float)
        q1 = np.nanpercentile(X_arr, 25, axis=0)
        q3 = np.nanpercentile(X_arr, 75, axis=0)
        iqr = q3 - q1

        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        """Clip input array to learned lower and upper bounds."""
        if self.lower_ is None or self.upper_ is None:
            raise NotFittedError("This IQRClipper instance is not fitted yet. Call 'fit' first.")

        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_, self.upper_)
