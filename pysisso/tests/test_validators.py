# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL, All rights reserved.
# Distributed open source for academic and non-profit users.
# Contact Matgenix for commercial usage.
# See LICENSE file for details.


import pathlib

import pytest
from monty.tempfile import ScratchDir

from pysisso.validators import NormalCompletionValidator


@pytest.mark.unit
def test_normal_completion_validator():
    v = NormalCompletionValidator(
        output_file="SISSO.out", stdout_file="SISSO.log", stderr_file="SISSO.err"
    )
    with ScratchDir("."):
        assert v.check() is True
    with ScratchDir("."):
        pathlib.Path("SISSO.err").touch()
        pathlib.Path("SISSO.log").write_text(data="something", encoding="utf-8")
        assert v.check() is True
    with ScratchDir("."):
        pathlib.Path("SISSO.err").touch()
        pathlib.Path("SISSO.out").write_text(
            data="Dummy line\n Have a nice day !\n", encoding="utf-8"
        )
        assert v.check() is True
    with ScratchDir("."):
        pathlib.Path("SISSO.err").touch()
        pathlib.Path("SISSO.log").touch()
        pathlib.Path("SISSO.out").write_text(
            data="Dummy line\n Have a nice day !\n", encoding="utf-8"
        )
        assert v.check() is True
    with ScratchDir("."):
        pathlib.Path("SISSO.err").write_text(data="Dummy error", encoding="utf-8")
        pathlib.Path("SISSO.log").write_text(data="something", encoding="utf-8")
        pathlib.Path("SISSO.out").write_text(
            data="Dummy line\n Have a nice day !\n", encoding="utf-8"
        )
        assert v.check() is True
    with ScratchDir("."):
        pathlib.Path("SISSO.err").touch()
        pathlib.Path("SISSO.log").write_text(data="something", encoding="utf-8")
        pathlib.Path("SISSO.out").write_text(
            data="Dummy line\n Have a nice day !\n", encoding="utf-8"
        )
        assert v.check() is False

    v = NormalCompletionValidator(
        output_file="mySISSO.out", stdout_file="mySISSO.log", stderr_file="mySISSO.err"
    )
    with ScratchDir("."):
        pathlib.Path("SISSO.err").touch()
        pathlib.Path("SISSO.log").write_text(data="something", encoding="utf-8")
        pathlib.Path("SISSO.out").write_text(
            data="Dummy line\n Have a nice day !\n", encoding="utf-8"
        )
        assert v.check() is True
    with ScratchDir("."):
        pathlib.Path("mySISSO.err").touch()
        pathlib.Path("mySISSO.log").write_text(data="something", encoding="utf-8")
        pathlib.Path("mySISSO.out").write_text(
            data="Dummy line\n Have a nice day !\n", encoding="utf-8"
        )
        assert v.check() is False
    with ScratchDir("."):
        pathlib.Path("mySISSO.log").write_text(data="something", encoding="utf-8")
        pathlib.Path("mySISSO.out").write_text(
            data="Dummy line\n Have a nice day !\n", encoding="utf-8"
        )
        assert v.check() is False
