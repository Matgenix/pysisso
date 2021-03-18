..
   Copyright (c) 2020, Matgenix SRL, All rights reserved.
   Distributed open source for academic and non-profit users.
   Contact Matgenix for commercial usage.
   See LICENSE file for details.


Release 0.3.2 (Mar 18, 2021)
============================

Features added
--------------

* Added OMP class method to set up Orthogonal Matching Pursuit parameters in
  SISSO regressor.

Miscellanous
------------

* Started documentation
* Updated spelling of "licence" to "license" everywhere


Release 0.3.1 (Feb 18, 2021)
============================

* Updated monty's versions requirement (>=3.0.4, <5)


Release 0.3 (Feb 17, 2021)
==========================

* Added LICENCE file
* Released pysisso under Matgenix's licence


Release 0.2.1 (Dec 14, 2020)
============================

Features added
--------------

* Modified SISSO output objects in order to be able to use model persistence in sklearn

  - Removed inner function that prevented to use joblib

* Added option for temporary run directory (useful for hyperparameter search and
  cross-validation)

* Added a list of basic examples of usage

* Fixed docstrings according to pydocstyle

Testing
-------

* Added pydocstyle in pre-commits

Release 0.2 (Dec 10, 2020)
==========================

Incompatible changes
--------------------

* Pysisso now relies on poetry and the pyproject.toml file for configuration and builds

  - Removed setup.py and setup.cfg
  - Installation with poetry using "poetry install" command
  - Distribution package (including setup.py generated automatically by poetry) using
    the "poetry build" command
  - See https://python-poetry.org/

Features added
--------------

* Added linting tools

  - pre-commit-hooks (trailing-whitespace, check-executables-have-shebangs,
    check-merge-conflict and name-tests-test)
  - bandit
  - flake8
  - black
  - isort

* Full test suite (unit and integration tests)

  - Based on pytest (see https://docs.pytest.org/en/stable/)
  - Using pytest-cov (using coverage, https://coverage.readthedocs.io/en/coverage-5.3/)
    to get code coverage reports (see https://pytest-cov.readthedocs.io/en/latest/)
  - Using pytest-mock (using unittest.mock,
    https://docs.python.org/3/library/unittest.mock.html) for mocking SISSO execution
    in unit tests (see https://github.com/pytest-dev/pytest-mock/)

* Added Github workflow for continuous integration with Github Actions

* All configuration has been defined in pyproject.toml (except for flake8)

* Miscellaneous (undocumented here) bug fixes

Release 0.1 (Aug 31, 2020)
==========================

Features
--------

* First release.

* Definition of inputs for SISSO

  - SISSO.in, some automatic checks of input arguments
  - .dat files

* Job to run SISSO using custodian

  - Validator for normal completion of SISSO

* Parsing of outputs of SISSO

  - Definition of SISSOModel objects containing the final expression determined by SISSO

* Scikit-learn compliant interface for regression with SISSO

  - fit method
  - predict method