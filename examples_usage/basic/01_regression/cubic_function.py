# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL, All rights reserved.
# Distributed open source for academic and non-profit users.
# Contact Matgenix for commercial usage.
# See LICENCE file for details.

"""Example usage of pysisso for a regression using sklearn interface."""

import matplotlib.pyplot as plt
import numpy as np

from pysisso.sklearn import SISSORegressor

# Define general parameters
TITLE = "f(x) = 0.5*x^3 + 0.5*x^2 - 4.0*x - 4.0"
NPOINTS = 100  # Number of data points
SIGMA = 0.5  # Randomness in the data points
PLOT_FIGURES = True  # whether to interactively plot the figures with matplotlib
SAVE_FIGURES = False  # whether to save the matplotlib figures to a file
CLEAN_RUN_DIR = True  # whether to remove the SISSO_dir after the fit

# Set the random seed to always keep the same figure
np.random.seed(42)


# Define the function:
# f(x) = 0.5*x^3 + 0.5*x^2 - 4.0*x - 4.0 (roots = [-2.0, -1.0, 2.0])
def fun(xx, const=1.0):
    return 0.5 * xx ** 3 + 0.5 * xx ** 2 - 4.0 * 0.5 * xx - 4.0 * const


# Define the data set
X = np.random.uniform(-2.5, 2.5, NPOINTS)
y = fun(X) + np.random.normal(0.0, scale=SIGMA, size=NPOINTS)

# Plot true function and data
xlin = np.linspace(-3, 3, 1000)
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

# Define the regressor, fit the data and predict
sisso_regressor = SISSORegressor(
    rung=1,
    opset="(+)(*)(^2)(^3)(^-1)(cos)(sin)",
    desc_dim=3,
    clean_run_dir=CLEAN_RUN_DIR,
)
X = X.reshape(-1, 1)  # only one feature, X is initially defined as 1D, sklearn needs 2D
sisso_regressor.fit(X, y)
ylin_pred = sisso_regressor.predict(xlin)

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
