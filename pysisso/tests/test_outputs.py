# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL, All rights reserved.
# Distributed open source for academic and non-profit users.
# Contact Matgenix for commercial usage.
# See LICENSE file for details.


import os

import numpy as np
import pandas as pd
import pytest

import pysisso
from pysisso.outputs import (
    SISSODescriptor,
    SISSOIteration,
    SISSOModel,
    SISSOOut,
    SISSOParams,
    SISSOVersion,
    scd,
)

TEST_FILES_DIR = os.path.abspath(
    os.path.join(pysisso.__file__, "..", "..", "test_files")
)

sisso_out = SISSOOut.from_file(
    filepath=os.path.join(TEST_FILES_DIR, "runs", "cubic_function", "SISSO.out")
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
    assert (
        str(sisso_params)
        == """Parameters for SISSO :
 - property_type : 3
 - descriptor_dimension : 3
 - total_number_properties : 1
 - task_weighting : [1]
 - number_of_samples : [100]
 - n_scalar_features : 1
 - n_rungs : 1
 - max_feature_complexity : 10
 - n_dimension_types : 0
 - dimension_types : [[]]
 - lower_bound_maxabs_value : 0.001
 - upper_bound_maxabs_value : 100000.0
 - SIS_subspaces_sizes : [20]
 - operators : ['(+)(*)(^2)(^3)(^-1)(cos)(sin)']
 - sparsification_method : L0
 - n_topmodels : 100
 - fit_intercept : True
 - metric : RMSE"""
    )
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
    assert str(descriptor_last_3) == descriptor_last_3.descriptor_string
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
    assert sisso_out.cpu_time == pytest.approx(0.64)
    models = sisso_out.models
    assert len(models) == 3
    assert isinstance(models[0], SISSOModel)
    assert isinstance(models[1], SISSOModel)
    assert isinstance(models[2], SISSOModel)

    # Partial SISSO output
    partial_sisso_out_fpath = os.path.join(
        TEST_FILES_DIR, "outputs", "SISSO.3.0.2.out_not_finished"
    )
    with pytest.raises(
        ValueError,
        match=r"Should get exactly one total " r"cpu time in the string, got 0.",
    ):
        SISSOOut.from_file(filepath=partial_sisso_out_fpath)
    sisso_out = SISSOOut.from_file(
        filepath=partial_sisso_out_fpath, allow_unfinished=True
    )
    assert len(sisso_out.iterations) == 2
    assert sisso_out.cpu_time is None
    models = sisso_out.models
    assert len(models) == 2
    assert isinstance(models[0], SISSOModel)
    assert isinstance(models[1], SISSOModel)


@pytest.mark.unit
def test_scd():
    assert scd(0.0) == pytest.approx(1.0 / np.pi)
    assert scd(1.0) == pytest.approx(0.5 / np.pi)
    assert scd(-1.0) == pytest.approx(0.5 / np.pi)
    assert scd(3.0) == pytest.approx(0.1 / np.pi)
    assert scd(-3.0) == pytest.approx(0.1 / np.pi)


@pytest.mark.unit
def test_decode_function():
    decoded = SISSODescriptor._decode_function("((myx)^3+sin(myx))")
    assert decoded["evalstring"] == "((df['myx'])**3+np.sin(df['myx']))"
    assert decoded["features_in_string"] == [
        {"featname": "myx", "istart": 2, "iend": 5},
        {"featname": "myx", "istart": 13, "iend": 16},
    ]
    assert decoded["inputs"] == ["myx"]

    with pytest.raises(ValueError, match=r'String should start and end with "#"'):
        SISSODescriptor._decode_function("tan(myx)")

    string = "((myx)^3+(sin(myx))^-1-((cos(a))^6+sin(b))^-1)"
    decoded = SISSODescriptor._decode_function(string)
    assert (
        decoded["evalstring"] == "((df['myx'])**3+1.0/(np.sin(df['myx']))"
        "-1.0/((np.cos(df['a']))**6+np.sin(df['b'])))"
    )
    assert decoded["features_in_string"] == [
        {"featname": "myx", "istart": 2, "iend": 5},
        {"featname": "myx", "istart": 14, "iend": 17},
        {"featname": "a", "istart": 29, "iend": 30},
        {"featname": "b", "istart": 39, "iend": 40},
    ]
    assert decoded["inputs"] == ["myx", "a", "b"]

    with pytest.raises(
        ValueError, match=r'Could not find initial parenthesis for "\)\^-1".'
    ):
        SISSODescriptor._decode_function("sin(myx))^-1")
