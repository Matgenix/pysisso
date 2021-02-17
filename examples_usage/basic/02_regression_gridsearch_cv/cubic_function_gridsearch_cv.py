# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL, All rights reserved.
# Distributed open source for academic and non-profit users.
# Contact Matgenix for commercial usage.
# See LICENCE file for details.

"""Example usage of pysisso for a regression using sklearn interface."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

from pysisso.sklearn import SISSORegressor

# Define general parameters
TITLE = "f(x) = 0.5*x^3 + 0.5*x^2 - 4.0*x - 4.0"
NPOINTS = 500  # Number of data points
SIGMA = 0.5  # Randomness in the data points
PLOT_FIGURES = True  # whether to interactively plot the figures with matplotlib
SAVE_FIGURES = False  # whether to save the matplotlib figures to a file
CLEAN_RUN_DIR = False  # whether to remove the SISSO_dir after the fit

# Set the random seed to always keep the same figure
np.random.seed(42)


# Define the function:
# f(x) = 0.5*x^3 + 0.5*x^2 - 4.0*x - 4.0 (roots = [-2.0, -1.0, 2.0])
def fun(xx, const=1.0):
    return 0.5 * xx ** 3 + 0.5 * xx ** 2 - 4.0 * 0.5 * xx - 4.0 * const


# Define the data set
X = np.random.uniform(-5, 5, NPOINTS)
y = fun(X) + np.random.normal(0.0, scale=SIGMA, size=NPOINTS)

# Plot true function and data
xlin = np.linspace(-6, 6, 1000)
ylin = fun(xlin)
fig, subplot = plt.subplots()
subplot.plot(xlin, ylin, "-", color="C0", label="True function")
subplot.plot(X, y, "o", color="C1", label="Data")
subplot.set_xlabel("x")
subplot.set_ylabel("f(x)")
subplot.set_title(TITLE)
subplot.legend()
if SAVE_FIGURES:
    fig.savefig("true_data.pdf")
if PLOT_FIGURES:
    plt.show()

# Define the regressor and the grid search, fit the data and predict
# Note that run_dir HAS to be None here, so that concurrent SISSO runs from different
#  folds and/or hyperparameters sets do not interfere with one another
sisso_regressor = SISSORegressor(
    rung=1,
    opset="(+)(-)(*)(^2)(^3)(^-1)(exp)(sin)(cos)",
    desc_dim=3,
    subs_sis=40,
    method="L1L0",
    L1L0_size4L0=15,
    run_dir=None,
    clean_run_dir=CLEAN_RUN_DIR,
)
param_grid = {"rung": [0, 1, 2], "desc_dim": [2, 3, 4]}
grid_search = GridSearchCV(
    estimator=sisso_regressor, param_grid=param_grid, cv=4, n_jobs=4
)
X = X.reshape(-1, 1)  # only one feature, X is initially defined as 1D, sklearn needs 2D
grid_search.fit(X, y)
ylin_pred = grid_search.predict(xlin)

# Plot the true and predicted functions, together with the data
fig, subplot = plt.subplots()
subplot.plot(xlin, ylin, "-", color="C0", label="True function")
subplot.plot(X, y, "o", color="C1", label="Data")
subplot.plot(xlin, ylin_pred, "-", color="C2", label="Predicted function")
subplot.set_xlabel("x")
subplot.set_ylabel("f(x)")
subplot.set_title(TITLE)
subplot.legend()
if SAVE_FIGURES:
    fig.savefig("true_data_pred.pdf")
if PLOT_FIGURES:
    plt.show()
