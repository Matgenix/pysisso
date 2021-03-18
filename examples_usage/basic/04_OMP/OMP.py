# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL, All rights reserved.
# Distributed open source for academic and non-profit users.
# Contact Matgenix for commercial usage.
# See LICENSE file for details.

"""Example usage of pysisso for a regression using sklearn interface with the
Orthogonal Matching Pursuit algorithm."""

import numpy as np

from pysisso.outputs import SISSOOut
from pysisso.sklearn import SISSORegressor

# Define the data set
X = np.array(
    [
        [8, 1, 3.01, 4],
        [6, 2, 3.02, 3],
        [2, 3, 3.01, 0],
        [10, 4, 3.02, -8],
        [4, 5, 3.01, 10],
    ]
)
y = 0.9 * X[:, 1] + 0.1 * X[:, 3] - 1.0

# Define the regressor and fit the data
sisso_reg = SISSORegressor.OMP(desc_dim=4)
sisso_reg.fit(X, y, columns=["feature_0", "feature_1", "feature_2", "feature_3"])

# Get the final model obtained
sisso_out = SISSOOut.from_file(filepath="SISSO_dir/SISSO.out")
sisso_model = sisso_out.model

# Get the descriptors
descriptors = [str(d) for d in sisso_model.descriptors]

# Print the order of the OMP features
# Should start with feature_1, then feature_3.
# feature_0 and feature_2 might be interchanged.
for idesc, desc in enumerate(descriptors):
    print(f"#{idesc+1}: {desc} ({sisso_model.coefficients[0][idesc]})")
