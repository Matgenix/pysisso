# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


from monty.tempfile import ScratchDir
import pathlib
from pysisso.jobs import SISSOJob
# from pysisso.utils import FakeExec
import pytest
import subprocess


# @pytest.mark.unit_test
def test_sisso_job():
    j = SISSOJob(SISSO_exe='fakeSISSO', nprocs=1, stdout_file='SISSO.log', stderr_file='SISSO.err')
    # with FakeExec('fakeSISSO', remove_temp_bin_dir=False):
    #     with ScratchDir('.', copy_to_current_on_exit=True):
    #         p = j.run()
    #         assert type(p) is subprocess.Popen
