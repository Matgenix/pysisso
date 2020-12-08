#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL

from setuptools import setup

setup(
    name="pysisso",
    version="0.0.1",
    description="Python interface to the SISSO (Sure Independence Screening and "
    "Sparsifying Operator) method.",
    author="David Waroquiers",
    author_email="david.waroquiers@matgenix.com",
    url="",
    packages=["pysisso"],
    install_requires=[
        "pandas>=1.0.5",
        "monty>=3.0.4",
        "custodian>=2018.8.10",
        "scikit-learn>=0.23.1",
    ],
    tests_require=["pytest", "pytest-cov", "pytest-mock"],
)
