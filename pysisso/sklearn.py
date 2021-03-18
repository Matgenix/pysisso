# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL, All rights reserved.
# Distributed open source for academic and non-profit users.
# Contact Matgenix for commercial usage.
# See LICENSE file for details.

"""Module containing a scikit-learn compliant interface to SISSO."""

import shutil
import tempfile
from datetime import datetime
from typing import Optional, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from custodian import Custodian  # type: ignore
from monty.os import cd, makedirs_p  # type: ignore
from sklearn.base import BaseEstimator, RegressorMixin  # type: ignore

from pysisso.inputs import SISSODat, SISSOIn
from pysisso.jobs import SISSOJob
from pysisso.outputs import SISSOOut


def get_timestamp(tstamp: Optional[datetime] = None) -> object:
    """Get a string representing the a time stamp.

    Args:
        tstamp: datetime.datetime object representing date and time. If set to None,
            the current time is taken.

    Returns:
        str: String representation of the time stamp.
    """
    tstamp = tstamp or datetime.now()
    return (
        f"{str(tstamp.year).zfill(4)}_{str(tstamp.month).zfill(2)}_"
        f"{str(tstamp.day).zfill(2)}_"
        f"{str(tstamp.hour).zfill(2)}_{str(tstamp.minute).zfill(2)}_"
        f"{str(tstamp.second).zfill(2)}_{str(tstamp.microsecond).zfill(6)}"
    )


class SISSORegressor(RegressorMixin, BaseEstimator):
    """SISSO regressor class compatible with scikit-learn."""

    def __init__(
        self,
        ntask=1,
        task_weighting=1,
        desc_dim=2,
        restart=False,
        rung=2,
        opset="(+)(-)",
        maxcomplexity=10,
        dimclass=None,
        maxfval_lb=1e-3,
        maxfval_ub=1e5,
        subs_sis=20,
        method="L0",
        L1L0_size4L0=1,
        fit_intercept=True,
        metric="RMSE",
        nm_output=100,
        isconvex=None,
        width=None,
        nvf=None,
        vfsize=None,
        vf2sf=None,
        npf_must=None,
        L1_max_iter=None,
        L1_tole=None,
        L1_dens=None,
        L1_nlambda=None,
        L1_minrmse=None,
        L1_warm_start=None,
        L1_weighted=None,
        features_dimensions: Union[dict, None] = None,
        use_custodian: bool = True,
        custodian_job_kwargs: Union[None, dict] = None,
        custodian_kwargs: Union[None, dict] = None,
        run_dir: Union[None, str] = "SISSO_dir",
        clean_run_dir: bool = False,
    ):  # noqa: D417
        """Construct SISSORegressor class.

        All arguments not listed below are arguments from the SISSO code. For more
        information, see https://github.com/rouyang2017/SISSO.

        Args:
            use_custodian: Whether to use custodian (currently mandatory).
            custodian_job_kwargs: Keyword arguments for custodian job.
            custodian_kwargs: Keyword arguments for custodian.
            run_dir: Name of the directory where SISSO is run. If None, the directory
                will be set automatically. It then contains a timestamp and is unique.
            clean_run_dir: Whether to clean the run directory after SISSO has run.
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
        self.L1L0_size4L0 = L1L0_size4L0  # pylint: disable=C0103
        self.fit_intercept = fit_intercept
        self.metric = metric
        self.nm_output = nm_output
        self.isconvex = isconvex
        self.width = width
        self.nvf = nvf
        self.vfsize = vfsize
        self.vf2sf = vf2sf
        self.npf_must = npf_must
        self.L1_max_iter = L1_max_iter  # pylint: disable=C0103
        self.L1_tole = L1_tole  # pylint: disable=C0103
        self.L1_dens = L1_dens  # pylint: disable=C0103
        self.L1_nlambda = L1_nlambda  # pylint: disable=C0103
        self.L1_minrmse = L1_minrmse  # pylint: disable=C0103
        self.L1_warm_start = L1_warm_start  # pylint: disable=C0103
        self.L1_weighted = L1_weighted  # pylint: disable=C0103
        self.features_dimensions = features_dimensions
        self.use_custodian = use_custodian
        self.custodian_job_kwargs = custodian_job_kwargs
        self.custodian_kwargs = custodian_kwargs
        self.run_dir = run_dir
        self.clean_run_dir = clean_run_dir

    def fit(self, X, y, index=None, columns=None, tasks=None):
        """Fit a SISSO regression based on inputs X and output y.

        This method supports Multi-Task SISSO. For Single-Task SISSO, y must have a
        shape (n_samples) or (n_samples, 1).
        For Multi-Task SISSO, y must have a shape (n_samples, n_tasks). The arrays
        will be reshaped to fit SISSO's input files.
        For example, with 10 samples and 3 properties, the output array (y) will be
        reshaped to (30, 1). The input array (X) is left unchanged.
        It is also possible to provide samples without an output for some properties
        by setting that property to NaN. In that case, the corresponding values in the
        input (X) and output (y) arrays will be removed from the SISSO inputs.
        In the previous example, if 2 of the samples have NaN for the first property,
        1 sample has Nan for the second property and 4 samples have Nan for the third
        property, the final output array (y) will have a shape (30-2-1-4, 1), i.e.
        (23, 1), while the final input array (X) will have a shape (23, n_features).

        Args:
            X: Feature vectors as an array-like of shape (n_samples, n_features).
            y: Target values as an array-like of shape (n_samples,)
                or (n_samples, n_tasks).
            index: List of string identifiers for each sample. If None, "sampleN"
                with N=[1, ..., n_samples] will be used.
            columns: List of string names of the features. If None, "featN"
                with N=[1, ..., n_features] will be used.
            tasks: When Multi-Task SISSO is used, this is the list of string names
                that will be used for each task/property. If None, "taskN"
                with N=[1, ..., n_tasks] will be used.
        """
        if not self.use_custodian:
            raise NotImplementedError

        self.sisso_in = SISSOIn.from_sisso_keywords(  # pylint: disable=W0201
            ptype=1,
            ntask=self.ntask,
            task_weighting=self.task_weighting,
            desc_dim=self.desc_dim,
            restart=self.restart,
            rung=self.rung,
            opset=self.opset,
            maxcomplexity=self.maxcomplexity,
            dimclass=self.dimclass,
            maxfval_lb=self.maxfval_lb,
            maxfval_ub=self.maxfval_ub,
            subs_sis=self.subs_sis,
            method=self.method,
            L1L0_size4L0=self.L1L0_size4L0,
            fit_intercept=self.fit_intercept,
            metric=self.metric,
            nm_output=self.nm_output,
            isconvex=self.isconvex,
            width=self.width,
            nvf=self.nvf,
            vfsize=self.vfsize,
            vf2sf=self.vf2sf,
            npf_must=self.npf_must,
            L1_max_iter=self.L1_max_iter,
            L1_tole=self.L1_tole,
            L1_dens=self.L1_dens,
            L1_nlambda=self.L1_nlambda,
            L1_minrmse=self.L1_minrmse,
            L1_warm_start=self.L1_warm_start,
            L1_weighted=self.L1_weighted,
        )
        # Set up columns. These columns are used by the SISSO model wrapper afterwards
        # for the prediction
        if columns is None and isinstance(X, pd.DataFrame):
            columns = list(X.columns)
        self.columns = columns or [  # pylint: disable=W0201
            "feat{:d}".format(ifeat) for ifeat in range(1, X.shape[1] + 1)
        ]
        if len(self.columns) != X.shape[1]:
            raise ValueError("Columns should be of the size of the second axis of X.")

        # Set up data
        X = np.array(X)
        y = np.array(y)
        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):  # Single-Task SISSO
            self.ntasks = 1  # pylint: disable=W0201
            index = index or [
                "sample{:d}".format(ii) for ii in range(1, X.shape[0] + 1)
            ]
            if len(index) != len(y) or len(index) != len(X):
                raise ValueError("Index, X and y should have same size.")
            nsample = None
        elif y.ndim == 2 and y.shape[1] > 1:  # Multi-Task SISSO
            self.ntasks = y.shape[1]  # pylint: disable=W0201
            samples_index = index or [
                "sample{:d}".format(ii) for ii in range(1, X.shape[0] + 1)
            ]
            tasks = tasks or ["task{:d}".format(ii) for ii in range(1, self.ntasks + 1)]
            newX = np.zeros((0, X.shape[1]))
            newy = np.array([])
            index = []
            nsample = []
            for itask in range(self.ntasks):
                yadd = y[:, itask]
                nanindices = np.argwhere(np.isnan(yadd)).flatten()
                totake = [ii for ii in range(len(yadd)) if ii not in nanindices]
                newy = np.concatenate([newy, np.take(yadd, indices=totake)])
                newX = np.row_stack([newX, np.take(X, indices=totake, axis=0)])
                nsample.append(len(totake))
                index.extend(
                    [
                        "{}_{}".format(sample_index, tasks[itask])
                        for i_sample, sample_index in enumerate(samples_index)
                        if i_sample in totake
                    ]
                )
            X = newX
            y = newy
        else:
            raise ValueError("Wrong shapes.")
        data = pd.DataFrame(X, index=index, columns=self.columns)
        data.insert(0, "target", y)
        data.insert(0, "identifier", index)

        # Set up SISSODat and SISSOIn
        sisso_dat = SISSODat(
            data=data, features_dimensions=self.features_dimensions, nsample=nsample
        )
        self.sisso_in.set_keywords_for_SISSO_dat(sisso_dat=sisso_dat)

        # Run SISSO
        if self.run_dir is None:
            makedirs_p("SISSO_runs")
            timestamp = get_timestamp()
            self.run_dir = tempfile.mkdtemp(
                suffix=None, prefix=f"SISSO_dir_{timestamp}_", dir="SISSO_runs"
            )
        else:
            makedirs_p(self.run_dir)
        with cd(self.run_dir):
            self.sisso_in.to_file(filename="SISSO.in")
            sisso_dat.to_file(filename="train.dat")
            job = SISSOJob()
            c = Custodian(jobs=[job], handlers=[], validators=[])
            c.run()
            self.sisso_out = SISSOOut.from_file(  # pylint: disable=W0201
                filepath="SISSO.out"
            )

        # Clean run directory
        if (
            self.clean_run_dir
        ):  # TODO: add check here to not remove "." if the user passes . ?
            shutil.rmtree(self.run_dir)

    def predict(self, X, index=None):
        """Predict output based on a fitted SISSO regression.

        Args:
            X: Feature vectors as an array-like of shape (n_samples, n_features).
            index: List of string identifiers for each sample. If None, "sampleN"
                with N=[1, ..., n_samples] will be used.
        """
        X = np.array(X)
        index = index or ["item{:d}".format(ii) for ii in range(X.shape[0])]
        data = pd.DataFrame(X, index=index, columns=self.columns)
        return self.sisso_out.model.predict(data)

    @classmethod
    def OMP(
        cls,
        desc_dim,
        use_custodian: bool = True,
        custodian_job_kwargs: Union[None, dict] = None,
        custodian_kwargs: Union[None, dict] = None,
        run_dir: Union[None, str] = "SISSO_dir",
        clean_run_dir: bool = False,
    ):
        """Construct SISSORegressor for Orthogonal Matching Pursuit (OMP).

        OMP is usually the first step to be performed before applying SISSO.
        Indeed, one starts with a relatively small set of base input descriptors
        (usually less than 20), that are then combined together by SISSO. One way to
        obtain this small set is to use the OMP algorithm (which is a particular case
        of the SISSO algorithm itself).

        Args:
            desc_dim: Number of descriptors to get with OMP.
            use_custodian: Whether to use custodian (currently mandatory).
            custodian_job_kwargs: Keyword arguments for custodian job.
            custodian_kwargs: Keyword arguments for custodian.
            run_dir: Name of the directory where SISSO is run. If None, the directory
                will be set automatically. It then contains a timestamp and is unique.
            clean_run_dir: Whether to clean the run directory after SISSO has run.

        Returns:
            SISSORegressor: SISSO regressor with OMP parameters.
        """
        return cls(
            opset="(+)(-)(*)(/)(exp)(exp-)(^-1)(^2)(^3)(sqrt)(cbrt)(log)(|-|)(scd)(^6)",
            rung=0,
            desc_dim=desc_dim,
            subs_sis=1,
            method="L0",
            L1L0_size4L0=None,
            features_dimensions=None,
            use_custodian=use_custodian,
            custodian_job_kwargs=custodian_job_kwargs,
            custodian_kwargs=custodian_kwargs,
            run_dir=run_dir,
            clean_run_dir=clean_run_dir,
        )

    @classmethod
    def from_SISSOIn(cls, sisso_in: SISSOIn):
        """Construct SISSORegressor from a SISSOIn object.

        Args:
            sisso_in: SISSOIn object containing the inputs for a SISSO run.

        Returns:
            SISSORegressor: SISSO regressor.
        """
        raise NotImplementedError
