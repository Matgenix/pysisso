# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


from pysisso.utils import get_version
from pysisso.utils import list_of_ints
from pysisso.utils import list_of_strs
from pysisso.utils import matrix_of_floats
from pysisso.utils import str_to_bool
import pytest


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


@pytest.mark.integration
def test_get_version():
    assert get_version() == {
        "version": (3, 0, 2),
        "header": "Version SISSO.3.0.2, June, 2020.",
    }
