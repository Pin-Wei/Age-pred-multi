#!/usr/bin/env python3


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TargetScaler(BaseEstimator, TransformerMixin):
    '''
    Scale multi-output targets relative to the first column.

    Notes
    -----
    - NaNs are ignored when estimating mean and std, 
      but are preserved in `transform` and `inverse_transform`.
    - `y` argument exists only for API compatibility.
    '''
    def fit(self, X: np.ndarray, y: np.ndarray = None):
        '''
        Estimate per-column nan-aware mean 
        and a scale derived from nan-aware standard deviations
        (expressed in units of column 0's std). 
        '''
        X = np.asarray(X, dtype=np.float64)
        std = np.nanstd(X, axis=0)
        std[std == 0] = 1.  # prevent division by zero for constant column(s)

        self.center_ = np.nanmean(X, axis=0)
        self.scale_ = std / std[0]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.center_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.center_
