# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


from monty.tempfile import ScratchDir
import os
from pysisso.jobs import SISSOJob
import pysisso
import pytest
import shutil
import subprocess


TEST_FILES_DIR = os.path.abspath(os.path.join(pysisso.__file__, '..', '..', 'test_files'))


@pytest.mark.integration
def test_sisso_job():
    j = SISSOJob(SISSO_exe='nonexistingSISSO', nprocs=1, stdout_file='SISSO.log', stderr_file='SISSO.err')
    with pytest.raises(ValueError, match='SISSOJob requires the SISSO executable to be in the path.\n'
                                         'Default executable name is "SISSO" and you provided "nonexistingSISSO".\n'
                                         'Download the SISSO code at https://github.com/rouyang2017/SISSO '
                                         'and compile the executable or fix the name of your executable.'):
        j.run()
    j = SISSOJob(SISSO_exe='SISSO', nprocs=1, stdout_file='SISSO.log', stderr_file='SISSO.err')
    with ScratchDir('.'):
        shutil.copy2(os.path.join(TEST_FILES_DIR, 'inputs', 'SISSO.in_regression'), 'SISSO.in')
        shutil.copy2(os.path.join(TEST_FILES_DIR, 'inputs', 'train.dat_regression'), 'train.dat')
        p = j.run()
        assert type(p) is subprocess.Popen
