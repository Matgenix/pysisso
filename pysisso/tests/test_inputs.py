# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


import os
from monty.tempfile import ScratchDir
from pysisso.inputs import SISSODat
from pysisso.inputs import SISSOIn
import pysisso
import pytest
import pandas as pd


TEST_FILES_DIR = os.path.abspath(
    os.path.join(pysisso.__file__, "..", "..", "test_files")
)


@pytest.mark.unit
def test_sisso_dat():
    sisso_dat = SISSODat.from_dat_file(
        filepath=os.path.join(TEST_FILES_DIR, "inputs", "train.dat_regression")
    )
    assert sisso_dat.nsample == 5
    assert sisso_dat.nsf == 3
    assert sisso_dat.ntask == 1
    assert isinstance(sisso_dat.data, pd.DataFrame)


@pytest.mark.unit
def test_sisso_in():
    sisso_dat = SISSODat.from_dat_file(
        filepath=os.path.join(TEST_FILES_DIR, "inputs", "train.dat_regression")
    )
    sisso_in = SISSOIn.from_SISSO_dat(sisso_dat=sisso_dat)
    assert sisso_in.is_regression is True
    with ScratchDir("."):
        sisso_in.to_file()
        assert os.path.exists("SISSO.in")
        with open("SISSO.in", "r") as f:
            content = f.read()
        assert "SISSO.in generated by Matgenix's pysisso package." in content
        assert "ntask=1\nnsample=5" in content
        assert "method='L0'\nL1L0_size4L0=1\nfit_intercept=.true." in content
