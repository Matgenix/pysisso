# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL, All rights reserved.
# Distributed open source for academic and non-profit users.
# Contact Matgenix for commercial usage.
# See LICENSE file for details.


import os
import shutil
import subprocess

import pytest
from monty.tempfile import ScratchDir

import pysisso.jobs
from pysisso.jobs import SISSOJob
from pysisso.utils import TEST_FILES_DIR


@pytest.mark.integration
def test_sisso_job():
    j = SISSOJob(
        SISSO_exe="nonexistingSISSO",
        nprocs=1,
        stdout_file="SISSO.log",
        stderr_file="SISSO.err",
    )
    with pytest.raises(
        ValueError,
        match="SISSOJob requires the SISSO executable to be in the path.\n"
        'Default executable name is "SISSO" and you provided "nonexistingSISSO".\n'
        "Download the SISSO code at https://github.com/rouyang2017/SISSO "
        "and compile the executable or fix the name of your executable.",
    ):
        j.run()
    j = SISSOJob(
        SISSO_exe="SISSO", nprocs=1, stdout_file="SISSO.log", stderr_file="SISSO.err"
    )
    with ScratchDir("."):
        shutil.copy2(
            os.path.join(TEST_FILES_DIR, "inputs", "SISSO.in_regression"), "SISSO.in"
        )
        shutil.copy2(
            os.path.join(TEST_FILES_DIR, "inputs", "train.dat_regression"), "train.dat"
        )
        p = j.run()
        assert type(p) is subprocess.Popen


@pytest.mark.unit
def test_sisso_job_unit(mocker):
    j = SISSOJob(
        SISSO_exe="SISSO",
        nprocs=1,
        stdout_file="SISSO.log",
        stderr_file="SISSO.err",
    )

    mocker.patch.object(
        pysisso.jobs,
        "which",
        return_value=[],
        side_effect=lambda x: None,
    )

    with pytest.raises(
        ValueError,
        match="SISSOJob requires the SISSO executable to be in the path.\n"
        'Default executable name is "SISSO" and you provided "SISSO".\n'
        "Download the SISSO code at https://github.com/rouyang2017/SISSO "
        "and compile the executable or fix the name of your executable.",
    ):
        j.run()

    mocker.patch.object(
        pysisso.jobs,
        "which",
        return_value=[],
        side_effect=lambda x: "echo",
    )

    j = SISSOJob(
        SISSO_exe="SISSO", nprocs=1, stdout_file="SISSO.log", stderr_file="SISSO.err"
    )

    with ScratchDir("."):
        p = j.run()
        assert type(p) is subprocess.Popen
