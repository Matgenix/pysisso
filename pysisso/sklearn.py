# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


from sklearn.base import BaseEstimator, RegressorMixin
from pysisso.inputs import SISSOIn
from pysisso.inputs import SISSODat
from pysisso.outputs import SISSOOut
from typing import Union
from monty.os import cd
from monty.os import makedirs_p
from custodian import Custodian
from pysisso.jobs import SISSOJob
import pandas as pd
import shutil
import numpy as np


class SISSORegressor(RegressorMixin, BaseEstimator):
    """SISSO regressor class compatible with scikit-learn."""

    def __init__(self, sisso_in: SISSOIn, features_dimensions: Union[dict, None]=None, use_custodian: bool=True,
                 custodian_job_kwargs: Union[None, dict]=None, custodian_kwargs: Union[None, dict]=None,
                 run_dir: str='SISSO_dir', clean_run_dir: bool=True):
        """Construct SISSORegressor class.

        Args:
            use_custodian:
        """
        self.sisso_in = sisso_in
        self.features_dimensions = features_dimensions
        self.use_custodian = use_custodian
        self.custodian_job_kwargs = custodian_job_kwargs
        self.custodian_kwargs = custodian_kwargs
        self.run_dir = run_dir
        self.clean_run_dir = clean_run_dir
        self.sisso_out = None

    def fit(self, X, y, index=None, columns=None):
        """Fit a SISSO regression based on inputs X and output y.

        Args:
            X: Feature vectors as an array-like of shape (n_samples, n_features).
            y: Target values as an array-like of shape (n_samples,).
            index:
            columns:
        """
        index = index or ['item{:d}'.format(ii) for ii in range(len(y))]
        if len(index) != len(y) or len(index) != len(X):
            raise ValueError('Index, X and y should have same size.')
        columns = columns or ['feat{:d}'.format(ifeat) for ifeat in range(1, len(X)+1)]
        if len(columns) != X.shape[1]:
            raise ValueError('Columns should be of the size of the second axis of X.')

        data = pd.DataFrame(X, index=index, columns=columns)
        data.insert(0, 'target', y)
        data.insert(0, 'identifier', index)
        sisso_dat = SISSODat(data=data, features_dimensions=self.features_dimensions)
        self.sisso_in.set_keywords_for_SISSO_dat(sisso_dat=sisso_dat)
        if not self.use_custodian:
            raise ValueError('Custodian is mandatory.')

        # Run SISSO
        makedirs_p(self.run_dir)
        with cd(self.run_dir):
            self.sisso_in.to_file(filename='SISSO.in')
            sisso_dat.to_file(filename='train.dat')
            job = SISSOJob()
            c = Custodian(jobs=[job], handlers=[], validators=[])
            c.run()
            self.sisso_out = SISSOOut.from_file(filename='SISSO.out')

        # Clean run directory
        # if self.clean_run_dir:
        #     shutil.rmtree(self.run_dir)

    def predict(self, X):
        """Predict output based on a fitted SISSO regression."""
        return self.sisso_out.model.evaluate(X)

    @classmethod
    def from_sisso_keywords(cls, use_custodian: bool=True, custodian_job_kwargs=None, custodian_kwargs=None,
                            **sissoin_kwargs):
        sissoin = SISSOIn.from_sisso_keywords(ptype=1, **sissoin_kwargs)
        return cls(sisso_in=sissoin, use_custodian=use_custodian,
                   custodian_job_kwargs=custodian_job_kwargs, custodian_kwargs=custodian_kwargs)


# # Define columns of the pandas DataFrame for the prediction of L*
# L_columns = ['IDENTITY', 'L*']  # SISSO can only learn one property at a time
# L_columns.extend(MM_FEATURES)
#
# # Create SISSO.in and train.dat
# L_sisso_dat = pysisso.inputs.SISSODat(FULL_DATA[L_columns], features_dimensions=MM_FEATURES_DIMENSIONS)
# L_sisso_in = pysisso.inputs.SISSOIn.from_SISSO_dat(L_sisso_dat, opset='(+)(-)(^2)(^-1)', rung=0, desc_dim=4)
# # L_sisso_in = pysisso.inputs.SISSOIn.from_SISSO_dat(L_sisso_dat, opset='(+)(-)(^2)(^-1)', rung=2, desc_dim=3)
#
# # Create directory of execution if needed
# SISSO_dir = 'SISSO_run'
# if not os.path.exists(SISSO_dir):
#     os.makedirs(SISSO_dir)
#
# # Run SISSO with pysisso
# with cd(SISSO_dir):
#     L_sisso_in.to_file(filename='SISSO.in')
#     L_sisso_dat.to_file(filename='train.dat')
#     job = pysisso.jobs.SISSOJob()
#     c = Custodian(handlers=[], jobs=[job])
#     c.run()
#
