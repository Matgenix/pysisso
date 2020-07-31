# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


from sklearn.base import BaseEstimator, RegressorMixin


class SISSORegressor(RegressorMixin, BaseEstimator):
    """SISSO regressor class compatible with scikit-learn."""

    def __init__(self, sisso_in, use_custodian: bool=True, custodian_job_kwargs=None, custodian_kwargs=None):
        """Construct SISSORegressor class.

        Args:
            use_custodian:
        """

    def fit(self, X, y):
        """Fit a SISSO regression based on inputs X and output y.

        Args:
            X:
            y:
        """

    def predict(self, X):
        """Predict output based on a fitted SISSO regression."""
