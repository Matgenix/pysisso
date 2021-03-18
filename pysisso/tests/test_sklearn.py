# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL, All rights reserved.
# Distributed open source for academic and non-profit users.
# Contact Matgenix for commercial usage.
# See LICENSE file for details.


import datetime
import os
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from monty.tempfile import ScratchDir

import pysisso
import pysisso.sklearn
from pysisso.outputs import SISSOOut
from pysisso.sklearn import SISSORegressor, get_timestamp

TEST_FILES_DIR = os.path.abspath(
    os.path.join(pysisso.__file__, "..", "..", "test_files")
)


@pytest.mark.unit
def test_get_timestamp(mocker):
    timestamp = get_timestamp(datetime.datetime(2014, 5, 28, 9, 6, 57, 6521))
    assert isinstance(timestamp, str)
    assert timestamp == "2014_05_28_09_06_57_006521"


@pytest.mark.unit
def test_sisso_regressor(mocker):

    # Simple single task SISSO runs with various options for the run directory
    # Mock the run of the custodian by just copying a reference SISSO.out file
    def copy_sisso_out():
        shutil.copy(
            os.path.join(TEST_FILES_DIR, "runs", "perfect_linear_5pts", "SISSO.out"),
            "SISSO.out",
        )

    mocker.patch.object(
        pysisso.sklearn.Custodian,
        "run",
        return_value=[],
        side_effect=copy_sisso_out,
    )

    makedirs_spy = mocker.spy(pysisso.sklearn, "makedirs_p")
    with ScratchDir("."):
        sisso_reg = SISSORegressor(desc_dim=1, rung=0, subs_sis=1, method="L0")
        sisso_reg.fit(np.array([[1], [2], [3], [4], [5]]), np.array([0, 1, 2, 3, 4]))
        pred = sisso_reg.predict([[1.5], [4.5]])
        assert pred[0] == 0.5
        assert pred[1] == 3.5
        assert os.path.exists("SISSO_dir")
        makedirs_spy.assert_called_with("SISSO_dir")
        assert makedirs_spy.call_count == 1
        makedirs_spy.reset_mock()

    with ScratchDir("."):
        sisso_reg = SISSORegressor(
            desc_dim=1, rung=0, subs_sis=1, method="L0", run_dir="mySISSOdir"
        )
        sisso_reg.fit(np.array([[1], [2], [3], [4], [5]]), np.array([0, 1, 2, 3, 4]))
        pred = sisso_reg.predict([[1.5], [4.5]])
        assert pred[0] == 0.5
        assert pred[1] == 3.5
        assert os.path.exists("mySISSOdir")
        assert not os.path.exists("SISSO_dir")
        makedirs_spy.assert_called_with("mySISSOdir")
        assert makedirs_spy.call_count == 1
        makedirs_spy.reset_mock()

    with ScratchDir("."):
        sisso_reg = SISSORegressor(
            desc_dim=1,
            rung=0,
            subs_sis=1,
            method="L0",
            run_dir="mySISSOdir",
            clean_run_dir=True,
        )
        sisso_reg.fit(np.array([[1], [2], [3], [4], [5]]), np.array([0, 1, 2, 3, 4]))
        pred = sisso_reg.predict([[1.5], [4.5]])
        assert pred[0] == 0.5
        assert pred[1] == 3.5
        assert not os.path.exists("mySISSOdir")
        makedirs_spy.assert_called_with("mySISSOdir")
        assert makedirs_spy.call_count == 1
        makedirs_spy.reset_mock()

    with ScratchDir("."):
        sisso_reg = SISSORegressor(
            desc_dim=1, rung=0, subs_sis=1, method="L0", clean_run_dir=True
        )
        sisso_reg.fit(np.array([[1], [2], [3], [4], [5]]), np.array([0, 1, 2, 3, 4]))
        pred = sisso_reg.predict([[1.5], [4.5]])
        assert pred[0] == 0.5
        assert pred[1] == 3.5
        assert not os.path.exists("SISSO_dir")
        makedirs_spy.assert_called_with("SISSO_dir")
        assert makedirs_spy.call_count == 1
        makedirs_spy.reset_mock()

    # Run with a temporary directory (i.e. when run_dir is None, useful for CV)
    # TODO : mocking tempfile did not work here for some reason ...
    mocker.patch(
        "pysisso.sklearn.get_timestamp",
        return_value="2018_09_28_16_04_54_017895",
    )
    with ScratchDir("."):
        sisso_reg = SISSORegressor(
            desc_dim=1,
            rung=0,
            subs_sis=1,
            method="L0",
            run_dir=None,
            clean_run_dir=False,
        )
        sisso_reg.fit(np.array([[1], [2], [3], [4], [5]]), np.array([0, 1, 2, 3, 4]))
        pred = sisso_reg.predict([[1.5], [4.5]])
        assert pred[0] == 0.5
        assert pred[1] == 3.5
        assert os.path.exists("SISSO_runs")
        dirs = os.listdir("SISSO_runs")
        assert len(dirs) == 1
        sisso_dir = dirs[0]
        assert sisso_dir.startswith("SISSO_dir_2018_09_28_16_04_54_017895_")
        makedirs_spy.assert_called_with("SISSO_runs")
        assert makedirs_spy.call_count == 1
        makedirs_spy.reset_mock()

    # Run with a temporary directory (i.e. when run_dir is None, useful for CV)
    with ScratchDir("."):
        sisso_reg = SISSORegressor(
            desc_dim=1,
            rung=0,
            subs_sis=1,
            method="L0",
            run_dir=None,
            clean_run_dir=True,
        )
        sisso_reg.fit(np.array([[1], [2], [3], [4], [5]]), np.array([0, 1, 2, 3, 4]))
        pred = sisso_reg.predict([[1.5], [4.5]])
        assert pred[0] == 0.5
        assert pred[1] == 3.5
        assert os.path.exists("SISSO_runs")
        dirs = os.listdir("SISSO_runs")
        assert len(dirs) == 0
        assert makedirs_spy.call_count == 1
        makedirs_spy.reset_mock()

    # Simple multi task SISSO run
    # Mock the run of the custodian by just copying a reference SISSO.out file
    def copy_sisso_out():
        shutil.copy(
            os.path.join(
                TEST_FILES_DIR, "runs", "perfect_linear_5pts_multi", "SISSO.out"
            ),
            "SISSO.out",
        )

    mocker.patch.object(
        pysisso.sklearn.Custodian,
        "run",
        return_value=[],
        side_effect=copy_sisso_out,
    )
    with ScratchDir("."):
        sisso_reg = SISSORegressor(desc_dim=1, rung=0, subs_sis=1, method="L0")
        sisso_reg.fit(
            np.array([[1], [2], [3], [4], [5]]),
            np.array([[0, -3], [1, -5], [2, -7], [3, -9], [4, -11]]),
        )
        pred = sisso_reg.predict([[1.5], [4.5]])
        assert pred[0] == pytest.approx([0.5, -4])
        assert pred[1] == pytest.approx([3.5, -10])
        assert sisso_reg.columns == ["feat1"]

    # Test of initializations and errors
    # Run with a numpy array
    with ScratchDir("."):
        sisso_reg = SISSORegressor(desc_dim=1, rung=0, subs_sis=1, method="L0")
        sisso_reg.fit(
            np.array([[1, 5], [2, 3], [3, 89], [4, 1], [5, 4]]),
            np.array([[0, -3], [1, -5], [2, -7], [3, -9], [4, -11]]),
        )
        assert sisso_reg.columns == ["feat1", "feat2"]

    # Run with a pandas Dataframe
    with ScratchDir("."):
        sisso_reg = SISSORegressor(desc_dim=1, rung=0, subs_sis=1, method="L0")
        X_df = pd.DataFrame(
            [[1, 5], [2, 3], [3, 89], [4, 1], [5, 4]], columns=["a", "b"]
        )
        sisso_reg.fit(X_df, np.array([[0, -3], [1, -5], [2, -7], [3, -9], [4, -11]]))
        assert sisso_reg.columns == ["a", "b"]

    # Run raising errors about columns
    with ScratchDir("."):
        sisso_reg = SISSORegressor(desc_dim=1, rung=0, subs_sis=1, method="L0")
        X_df = pd.DataFrame(
            [[1, 5], [2, 3], [3, 89], [4, 1], [5, 4]], columns=["a", "b"]
        )
        with pytest.raises(
            ValueError,
            match=r"Columns should be of the size of the " r"second axis of X.",
        ):
            sisso_reg.fit(
                X_df,
                np.array([[0, -3], [1, -5], [2, -7], [3, -9], [4, -11]]),
                columns=["a", "b", "c"],
            )

    # Run raising errors about index
    with ScratchDir("."):
        sisso_reg = SISSORegressor(desc_dim=1, rung=0, subs_sis=1, method="L0")
        X_df = pd.DataFrame([[1], [2], [3], [4], [5]])
        with pytest.raises(ValueError, match=r"Index, X and y should have same size."):
            sisso_reg.fit(X_df, np.array([[0], [1], [2], [3]]))

    # Run raising errors about index
    with ScratchDir("."):
        sisso_reg = SISSORegressor(desc_dim=1, rung=0, subs_sis=1, method="L0")
        X_df = pd.DataFrame([[1], [2], [3], [4], [5]])
        with pytest.raises(ValueError, match=r"Index, X and y should have same size."):
            sisso_reg.fit(
                X_df,
                np.array([[0], [1], [2], [3], [4]]),
                index=["a", "b", "c", "d", "e", "f"],
            )

    # Run with a wrong shape for y target
    with ScratchDir("."):
        sisso_reg = SISSORegressor(desc_dim=1, rung=0, subs_sis=1, method="L0")
        X_df = pd.DataFrame([[1], [2], [3], [4], [5]])
        with pytest.raises(ValueError, match=r"Wrong shapes."):
            sisso_reg.fit(X_df, np.array([[[0], [1], [2], [3], [4]]]))


@pytest.mark.unit
def test_sisso_regressor_omp(mocker):
    # Simple SISSO run with OMP
    # Mock the run of the custodian by just copying a reference SISSO.out file
    def copy_sisso_out():
        shutil.copy(
            os.path.join(TEST_FILES_DIR, "runs", "OMP", "SISSO.out"),
            "SISSO.out",
        )

    mocker.patch.object(
        pysisso.sklearn.Custodian,
        "run",
        return_value=[],
        side_effect=copy_sisso_out,
    )
    with ScratchDir("."):
        sisso_reg = SISSORegressor.OMP(desc_dim=4)
        assert sisso_reg.rung == 0
        assert sisso_reg.subs_sis == 1
        assert sisso_reg.desc_dim == 4
        assert sisso_reg.method == "L0"
        assert sisso_reg.L1L0_size4L0 is None
        X = np.array(
            [
                [8, 1, 3.01, 4],
                [6, 2, 3.02, 3],
                [2, 3, 3.01, 0],
                [10, 4, 3.02, -8],
                [4, 5, 3.01, 10],
            ]
        )
        y = 0.9 * X[:, 1] + 0.1 * X[:, 3] - 1.0
        sisso_reg.fit(X, y)

        actual_sin = "SISSO_dir/SISSO.in"
        ref_sin = os.path.join(TEST_FILES_DIR, "runs", "OMP", "SISSO.in")
        assert [line for line in open(actual_sin)] == [line for line in open(ref_sin)]

        sisso_out = SISSOOut.from_file(filepath="SISSO_dir/SISSO.out")
        assert sisso_out.params.n_rungs == sisso_reg.rung
        assert sisso_out.params.SIS_subspaces_sizes == [sisso_reg.subs_sis]
        assert sisso_out.params.descriptor_dimension == sisso_reg.desc_dim
        assert sisso_out.params.sparsification_method == sisso_reg.method

        sisso_model = sisso_out.model
        assert str(sisso_model.descriptors[0]) == "(feature_1)"
        assert str(sisso_model.descriptors[1]) == "(feature_3)"


@pytest.mark.unit
def test_model_persistence(mocker):
    # Simple single task SISSO runs with various options for the run directory
    # Mock the run of the custodian by just copying a reference SISSO.out file
    def copy_sisso_out():
        shutil.copy(
            os.path.join(TEST_FILES_DIR, "runs", "perfect_linear_5pts", "SISSO.out"),
            "SISSO.out",
        )

    mocker.patch.object(
        pysisso.sklearn.Custodian,
        "run",
        return_value=[],
        side_effect=copy_sisso_out,
    )

    with ScratchDir("."):
        sisso_reg = SISSORegressor(desc_dim=1, rung=0, subs_sis=1, method="L0")
        sisso_reg.fit(np.array([[1], [2], [3], [4], [5]]), np.array([0, 1, 2, 3, 4]))
        joblib.dump(sisso_reg, filename="model.joblib")
        sisso_reg_loaded = joblib.load("model.joblib")
        pred = sisso_reg_loaded.predict([[1.5], [4.5]])
        assert pred[0] == 0.5
        assert pred[1] == 3.5
        assert sisso_reg.get_params() == sisso_reg_loaded.get_params()
        model = sisso_reg.sisso_out.model
        model_loaded = sisso_reg_loaded.sisso_out.model
        assert len(model.coefficients) == 1
        assert len(model_loaded.coefficients) == 1
        assert model.coefficients[0] == pytest.approx(model_loaded.coefficients[0])
