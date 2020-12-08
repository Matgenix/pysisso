# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


import os
from monty.tempfile import ScratchDir
import pysisso
import shutil
import subprocess
from typing import List
from typing import Union


TEST_FILES_DIR = os.path.abspath(
    os.path.join(pysisso.__file__, "..", "..", "test_files")
)


def get_version(SISSO_exe="SISSO"):
    # TODO: check how SISSO<3.0.2 was working
    with ScratchDir("."):
        shutil.copy2(
            os.path.join(TEST_FILES_DIR, "inputs", "SISSO.in_simple"), "SISSO.in"
        )
        shutil.copy2(
            os.path.join(TEST_FILES_DIR, "inputs", "train.dat_regression"), "train.dat"
        )
        with open("SISSO.log", "w") as f_stdout, open(
            "SISSO.err", "w", buffering=1
        ) as f_stderr:
            subprocess.call([SISSO_exe], stdin=None, stdout=f_stdout, stderr=f_stderr)
            with open("SISSO.out", "r") as f:
                header = f.readline()
                if (
                    "Version" not in header
                ):  # pragma: no cover # Reason: unlikely error of pysisso.
                    raise ValueError("Could not determine SISSO version.")
                version = tuple(
                    [int(ii) for ii in header.split(",")[0].split(".")[1:4]]
                )
                return {"version": version, "header": header.strip()}


def list_of_ints(string: str, delimiter: Union[str, None] = None) -> List[int]:
    """Cast a string to a list of integers.

    Args:
        string: String to be converted to a list of int's.
        delimiter: Delimiter between integers in the string.
            Default is to split with any whitespace string (see str.split() method).
    """

    return [int(sp) for sp in string.split(sep=delimiter)]


def list_of_strs(
    string: str, delimiter: Union[str, None] = None, strip=True
) -> List[str]:
    """Cast a string to a list of strings.

    Args:
        string: String to be converted to a list of str's.
        delimiter: Delimiter between str's in the string.
            Default is to split with any whitespace string (see str.split() method).
        strip: Whether to strip the substrings (i.e. remove leading and trailing whitespaces after the split with
            a delimiter that is not whitespace)
    """

    if strip:
        return [s.strip() for s in string.split(sep=delimiter)]
    return string.split(sep=delimiter)


def matrix_of_floats(
    string: str, delimiter_ax0: str = "\n", delimiter_ax1: Union[str, None] = None
) -> List[List[float]]:
    """Cast a string to a list of list of floats.

    Args:
        string: String to be converted to a list of lists of floats.
        delimiter_ax0: Delimiter for the first axis of the matrix.
        delimiter_ax1: Delimiter for the second axis of the matrix.
    """

    return [
        [float(sp2) for sp2 in sp.split(sep=delimiter_ax1)]
        for sp in string.split(sep=delimiter_ax0)
    ]


def str_to_bool(string: str) -> bool:
    """Cast a string to a boolean value.

    Args:
        string: String to be converted to a bool.

    Raises:
        ValueError: In case the string could not be converted to a bool.
    """
    strip = string.strip()
    if strip in [".True.", "True", "T", "true", ".true."]:
        return True
    elif strip in [".False.", "False", "F", "false", ".false."]:
        return False
    raise ValueError('Could not convert "{}" to a boolean.'.format(string))


# class FakeExec:
#
#     def __init__(self, exec_name, remove_temp_bin_dir=True):
#         self.exec_name = exec_name
#         self.remove_temp_bin_dir = remove_temp_bin_dir
#
#     def __enter__(self):
#         self.temp_bin_dir = tempfile.mkdtemp(prefix='bin', dir='.')
#         # Create fake executable file and make it executable (equivalent of "chmod +x")
#         exec_path = os.path.join(self.temp_bin_dir, self.exec_name)
#         print(exec_path)
#         with open(exec_path, 'w') as f:
#             f.write('#! /usr/bin/env python\npass\n')
#         st = os.stat(exec_path)
#         os.chmod(exec_path, st.st_mode | stat.S_IEXEC)
#
#         # Update PATH
#         self.current_OS_PATH = os.environ.get('PATH', None)
#         if self.current_OS_PATH is None:
#             paths = []
#         else:
#             paths = self.current_OS_PATH.split(':')
#         paths.insert(0, self.temp_bin_dir)
#         os.environ['PATH'] = ':'.join(paths)
#
#         return exec_path
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # Remove temporary bin directory with fake executable from system's PATH environment variable
#         OS_PATH = os.environ['PATH']
#         paths = OS_PATH.split(':')
#         paths.remove(self.temp_bin_dir)
#         os.environ['PATH'] = ':'.join(paths)
#
#         if self.remove_temp_bin_dir:
#             monty.shutil.remove(self.temp_bin_dir)
