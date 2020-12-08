# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


import os
import shutil

import numpy as np
import pytest
from monty.tempfile import ScratchDir

import pysisso
import pysisso.sklearn
from pysisso.sklearn import SISSORegressor

TEST_FILES_DIR = os.path.abspath(
    os.path.join(pysisso.__file__, "..", "..", "test_files")
)


@pytest.mark.unit
def test_sisso_regressor(mocker):
    with ScratchDir("."):

        def copy_sisso_out():
            shutil.copy(
                os.path.join(
                    TEST_FILES_DIR, "runs", "perfect_linear_5pts", "SISSO.out"
                ),
                "SISSO.out",
            )

        mocker.patch.object(
            pysisso.sklearn.Custodian,
            "run",
            return_value=[],
            side_effect=copy_sisso_out,
        )
        sisso_reg = SISSORegressor(desc_dim=1, rung=0, subs_sis=1, method="L0")
        sisso_reg.fit(np.array([[1], [2], [3], [4], [5]]), np.array([0, 1, 2, 3, 4]))
        pred = sisso_reg.predict([[1.5], [4.5]])
        assert pred[0] == 0.5
        assert pred[1] == 3.5
