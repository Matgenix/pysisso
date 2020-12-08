# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


import os

import pandas as pd
import pytest

import pysisso
from pysisso.outputs import (
    SISSODescriptor,
    SISSOIteration,
    SISSOOut,
    SISSOParams,
    SISSOVersion,
)

TEST_FILES_DIR = os.path.abspath(
    os.path.join(pysisso.__file__, "..", "..", "test_files")
)


@pytest.mark.unit
def test_sisso_out():
    sisso_out = SISSOOut.from_file(
        filepath=os.path.join(TEST_FILES_DIR, "runs", "cubic_function", "SISSO.out")
    )

    sisso_version = sisso_out.version
    assert isinstance(sisso_version, SISSOVersion)
    assert sisso_version.version == (3, 0, 2)
    assert sisso_version.header_string == "Version SISSO.3.0.2, June, 2020."
    sisso_params = sisso_out.params
    assert isinstance(sisso_params, SISSOParams)
    assert sisso_params.number_of_samples == [100]
    assert sisso_params.sparsification_method == "L0"
    sisso_iterations = sisso_out.iterations
    assert isinstance(sisso_iterations, list)
    assert len(sisso_iterations) == sisso_params.descriptor_dimension
    iteration_1 = sisso_iterations[0]
    last_iteration = sisso_iterations[-1]
    assert isinstance(iteration_1, SISSOIteration)
    assert isinstance(last_iteration, SISSOIteration)
    assert len(iteration_1.sisso_model.descriptors) == 1
    assert (
        len(last_iteration.sisso_model.descriptors) == sisso_params.descriptor_dimension
    )
    assert iteration_1.iteration_number == 1
    assert last_iteration.iteration_number == 3
    assert iteration_1.SIS_subspace_size == 6
    assert last_iteration.SIS_subspace_size == 0
    model_1 = iteration_1.sisso_model
    last_model = last_iteration.sisso_model
    assert model_1.dimension == 1
    assert last_model.dimension == 3
    assert len(model_1.descriptors) == 1
    assert len(last_model.descriptors) == 3
    assert len(model_1.rmse) == 1
    assert len(model_1.maxae) == 1
    assert len(last_model.rmse) == 1
    assert len(last_model.maxae) == 1
    assert model_1.rmse[0] == pytest.approx(0.7959386860e01)
    assert model_1.maxae[0] == pytest.approx(0.1858248525e02)
    assert last_model.rmse[0] == pytest.approx(0.1757799850e01)
    assert last_model.maxae[0] == pytest.approx(0.4267977958e01)
    assert len(model_1.coefficients) == 1
    assert len(last_model.coefficients) == 1
    assert len(model_1.coefficients[0]) == 1
    assert len(last_model.coefficients[0]) == 3
    assert model_1.coefficients[0] == pytest.approx([0.2553319133e00])
    assert last_model.coefficients[0] == pytest.approx(
        [0.9856312325e00, -0.3842863966e01, -0.1417565675e01]
    )
    assert len(model_1.intercept) == 1
    assert len(last_model.intercept) == 1
    assert model_1.intercept[0] == pytest.approx(-0.5364436924e01)
    assert last_model.intercept[0] == pytest.approx(0.3890294191e01)
    descriptors_1 = model_1.descriptors
    assert len(descriptors_1) == 1
    descriptors_last = last_model.descriptors
    assert len(descriptors_last) == 3
    descriptor_1 = descriptors_1[0]
    assert isinstance(descriptor_1, SISSODescriptor)
    assert descriptor_1.descriptor_id == 1
    assert descriptor_1.descriptor_string == "(myx)^3"
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["XX", "myx", "ZZ"])
    descr_1_eval = descriptor_1.evaluate(df)
    assert len(descr_1_eval) == 2
    assert descr_1_eval[0] == pytest.approx(8)
    assert descr_1_eval[1] == pytest.approx(125)
    descriptor_last_1 = descriptors_last[0]
    descriptor_last_2 = descriptors_last[1]
    descriptor_last_3 = descriptors_last[2]
    assert descriptor_last_1.descriptor_id == 1
    assert descriptor_last_2.descriptor_id == 2
    assert descriptor_last_3.descriptor_id == 3
    assert descriptor_last_1.descriptor_string == "(myx)^3"
    assert descriptor_last_2.descriptor_string == "(myx)^2"
    assert descriptor_last_3.descriptor_string == "(myx)"
    descr_last_1_eval = descriptor_last_1.evaluate(df)
    descr_last_2_eval = descriptor_last_2.evaluate(df)
    descr_last_3_eval = descriptor_last_3.evaluate(df)
    assert descr_last_1_eval[0] == pytest.approx(8)
    assert descr_last_1_eval[1] == pytest.approx(125)
    assert descr_last_2_eval[0] == pytest.approx(4)
    assert descr_last_2_eval[1] == pytest.approx(25)
    assert descr_last_3_eval[0] == pytest.approx(2)
    assert descr_last_3_eval[1] == pytest.approx(5)
    pred_1 = model_1.predict(df)
    assert pred_1[0] == pytest.approx(-3.3217816175999997)
    assert pred_1[1] == pytest.approx(26.5520522385)
    pred_last = last_model.predict(df)
    assert pred_last[0] == pytest.approx(-6.431243163)
    assert pred_last[1] == pytest.approx(23.9347707285)
