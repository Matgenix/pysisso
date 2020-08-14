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

    def __init__(self,
                 ntask=1, task_weighting=1, desc_dim=2, restart=False,
                 rung=2, opset='(+)(-)', maxcomplexity=10, dimclass=None,
                 maxfval_lb=1e-3, maxfval_ub=1e5, subs_sis=20,
                 method='L0', L1L0_size4L0=1, fit_intercept=True, metric='RMSE', nm_output=100,
                 isconvex=None, width=None, nvf=None, vfsize=None, vf2sf=None, npf_must=None,
                 L1_max_iter=None, L1_tole=None, L1_dens=None, L1_nlambda=None, L1_minrmse=None,
                 L1_warm_start=None, L1_weighted=None,
                 features_dimensions: Union[dict, None]=None, use_custodian: bool=True,
                 custodian_job_kwargs: Union[None, dict]=None, custodian_kwargs: Union[None, dict]=None,
                 run_dir: str='SISSO_dir', clean_run_dir: bool=False):
        """Construct SISSORegressor class.

        Args:
            use_custodian:
        """
        self.ntask = ntask
        self.task_weighting = task_weighting
        self.desc_dim = desc_dim
        self.restart = restart
        self.rung = rung
        self.opset = opset
        self.maxcomplexity = maxcomplexity
        self.dimclass = dimclass
        self.maxfval_lb = maxfval_lb
        self.maxfval_ub = maxfval_ub
        self.subs_sis = subs_sis
        self.method = method
        self.L1L0_size4L0 = L1L0_size4L0
        self.fit_intercept = fit_intercept
        self.metric = metric
        self.nm_output = nm_output
        self.isconvex = isconvex
        self.width = width
        self.nvf = nvf
        self.vfsize = vfsize
        self.vf2sf = vf2sf
        self.npf_must = npf_must
        self.L1_max_iter = L1_max_iter
        self.L1_tole = L1_tole
        self.L1_dens = L1_dens
        self.L1_nlambda = L1_nlambda
        self.L1_minrmse = L1_minrmse
        self.L1_warm_start = L1_warm_start
        self.L1_weighted = L1_weighted
        self.features_dimensions = features_dimensions
        self.use_custodian = use_custodian
        self.custodian_job_kwargs = custodian_job_kwargs
        self.custodian_kwargs = custodian_kwargs
        self.run_dir = run_dir
        self.clean_run_dir = clean_run_dir

    def fit(self, X, y, index=None, columns=None):
        """Fit a SISSO regression based on inputs X and output y.

        Args:
            X: Feature vectors as an array-like of shape (n_samples, n_features).
            y: Target values as an array-like of shape (n_samples,).
            index:
            columns:
        """
        self.sisso_in = SISSOIn.from_sisso_keywords(ptype=1, ntask=self.ntask,
                                                    task_weighting=self.task_weighting, desc_dim=self.desc_dim, restart=self.restart,
                                                    rung=self.rung, opset=self.opset, maxcomplexity=self.maxcomplexity,
                                                    dimclass=self.dimclass,
                                                    maxfval_lb=self.maxfval_lb, maxfval_ub=self.maxfval_ub, subs_sis=self.subs_sis,
                                                    method=self.method, L1L0_size4L0=self.L1L0_size4L0,
                                                    fit_intercept=self.fit_intercept, metric=self.metric, nm_output=self.nm_output,
                                                    isconvex=self.isconvex, width=self.width, nvf=self.nvf, vfsize=self.vfsize,
                                                    vf2sf=self.vf2sf, npf_must=self.npf_must,
                                                    L1_max_iter=self.L1_max_iter, L1_tole=self.L1_tole, L1_dens=self.L1_dens,
                                                    L1_nlambda=self.L1_nlambda, L1_minrmse=self.L1_minrmse,
                                                    L1_warm_start=self.L1_warm_start, L1_weighted=self.L1_weighted)
        if columns is None and isinstance(X, pd.DataFrame):
            columns = list(X.columns)
        X = np.array(X)
        y = np.array(y)
        index = index or ['item{:d}'.format(ii) for ii in range(X.shape[0])]
        if len(index) != len(y) or len(index) != len(X):
            raise ValueError('Index, X and y should have same size.')
        self.columns = columns or ['feat{:d}'.format(ifeat) for ifeat in range(1, X.shape[1]+1)]
        if len(self.columns) != X.shape[1]:
            raise ValueError('Columns should be of the size of the second axis of X.')

        data = pd.DataFrame(X, index=index, columns=self.columns)
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
        if self.clean_run_dir:
            shutil.rmtree(self.run_dir)

    def predict(self, X, index=None):
        """Predict output based on a fitted SISSO regression."""
        X = np.array(X)
        index = index or ['item{:d}'.format(ii) for ii in range(X.shape[0])]
        data = pd.DataFrame(X, index=index, columns=self.columns)
        return self.sisso_out.model.predict(data)

    # @classmethod
    # def from_sisso_keywords(cls, use_custodian: bool=True, custodian_job_kwargs=None, custodian_kwargs=None,
    #                         **sissoin_kwargs):
    #     sissoin = SISSOIn.from_sisso_keywords(ptype=1, **sissoin_kwargs)
    #     return cls(sisso_in=sissoin, use_custodian=use_custodian,
    #                custodian_job_kwargs=custodian_job_kwargs, custodian_kwargs=custodian_kwargs)
    #
    @classmethod
    def from_SISSOIn(cls, sisso_in: SISSOIn):
        pass