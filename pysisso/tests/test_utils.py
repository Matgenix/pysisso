# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL, All rights reserved.
# Distributed open source for academic and non-profit users.
# Contact Matgenix for commercial usage.
# See LICENSE file for details.


import os
import shutil

import pytest

import pysisso
from pysisso.utils import (
    get_version,
    list_of_ints,
    list_of_strs,
    matrix_of_floats,
    str_to_bool,
    subprocess,
)

TEST_FILES_DIR = os.path.abspath(
    os.path.join(pysisso.__file__, "..", "..", "test_files")
)


@pytest.mark.unit
def test_list_of_ints():
    assert list_of_ints(" 3 5   8") == [3, 5, 8]
    assert list_of_ints(" -1, 2,   4", delimiter=",") == [-1, 2, 4]


@pytest.mark.unit
def test_list_of_strs():
    assert list_of_strs(" 3 5   8") == ["3", "5", "8"]
    assert list_of_strs(" -1, 2 ,   4 ", delimiter=",") == ["-1", "2", "4"]
    assert list_of_strs(" -1, 2 ,   4 ", delimiter=",", strip=False) == [
        " -1",
        " 2 ",
        "   4 ",
    ]


@pytest.mark.unit
def test_matrix_of_floats():
    mf = matrix_of_floats(" 1 2   3\n4 5.3 -6.1")
    assert mf[0] == pytest.approx([1, 2, 3])
    assert mf[1] == pytest.approx([4, 5.3, -6.1])
    mf = matrix_of_floats(" 1 2   3 | 4 5.3 -6.1", delimiter_ax0="|")
    assert mf[0] == pytest.approx([1, 2, 3])
    assert mf[1] == pytest.approx([4, 5.3, -6.1])
    mf = matrix_of_floats(
        " 1, 2  , 3 | 4, 5.3 ,-6.1", delimiter_ax0="|", delimiter_ax1=","
    )
    assert mf[0] == pytest.approx([1, 2, 3])
    assert mf[1] == pytest.approx([4, 5.3, -6.1])
    mf = matrix_of_floats(" 1, 2  , 3 \n 4, 5.3 ,-6.1", delimiter_ax1=",")
    assert mf[0] == pytest.approx([1, 2, 3])
    assert mf[1] == pytest.approx([4, 5.3, -6.1])


@pytest.mark.unit
def test_str_to_bool():
    assert str_to_bool(".True.") is True
    assert str_to_bool("True") is True
    assert str_to_bool("T") is True
    assert str_to_bool("true") is True
    assert str_to_bool(".true.") is True
    assert str_to_bool(".False.") is False
    assert str_to_bool("False") is False
    assert str_to_bool("F") is False
    assert str_to_bool("false") is False
    assert str_to_bool(".false.") is False
    with pytest.raises(ValueError, match='Could not convert "t" to a boolean.'):
        str_to_bool("t")
    with pytest.raises(ValueError, match='Could not convert "falsy" to a boolean.'):
        str_to_bool("falsy")


@pytest.mark.unit
def test_get_version(mocker):
    def copy_sisso_out(*_, **__):
        shutil.copy(
            os.path.join(TEST_FILES_DIR, "outputs", "SISSO.3.0.2.out"),
            "SISSO.out",
        )

    mocker.patch.object(
        subprocess,
        "call",
        return_value=[],
        side_effect=copy_sisso_out,
    )
    assert get_version() == {
        "version": (3, 0, 2),
        "header": "Version SISSO.3.0.2, June, 2020.",
    }
