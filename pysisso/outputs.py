# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL, All rights reserved.
# Distributed open source for academic and non-profit users.
# Contact Matgenix for commercial usage.
# See LICENSE file for details.

"""Module containing classes to parse SISSO output files."""

import re
from typing import Callable, List, Mapping, Optional, Tuple, Union

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from monty.json import MSONable  # type: ignore

from pysisso.utils import list_of_ints, list_of_strs, matrix_of_floats, str_to_bool


class SISSOVersion(MSONable):
    """Class containing information about the SISSO version used."""

    def __init__(self, header_string: str, version: Tuple[int, int, int]):
        """Construct SISSOVersion class.

        Args:
            header_string: Header string found in the SISSO.out output file.
            version: Version of SISSO extracted from the header string.
        """
        self.header_string = header_string
        self.version = version

    @classmethod
    def from_string(cls, string: str):
        """Construct SISSOVersion from string.

        Args:
            string: First line from the SISSO.out output file.
        """
        version_sp = string.split(",")[0].split(".")
        return cls(
            header_string=string.strip(),
            version=(int(version_sp[1]), int(version_sp[2]), int(version_sp[3])),
        )


def scd(x):
    """Get Standard Cauchy Distribution of x.

    The Standard Cauchy Distribution (SCD) of x is :

    SCD(x) = (1.0 / pi) * 1.0 / (1.0 + x^2)

    Args:
        x: Value(s) for which the Standard Cauchy Distribution is needed.

    Returns:
        Standard Cauchy Distribution at value(s) x.
    """
    return 1.0 / (np.pi * (1.0 + x * x))


class SISSODescriptor(MSONable):
    """Class containing one composed descriptor."""

    def __init__(self, descriptor_id: int, descriptor_string: str):
        """Construct SISSODescriptor class.

        Args:
            descriptor_id: Integer identifier of this descriptor.
            descriptor_string: String description of this descriptor.
        """
        self.descriptor_id = descriptor_id
        self.descriptor_string = descriptor_string
        self.evalstring = self._decode_function(self.descriptor_string)["evalstring"]

    def evaluate(self, df):  # pylint: disable=W0613
        """Evaluate the descriptor from a given Dataframe.

        Args:
            df: panda's Dataframe to evaluate SISSODescriptor

        Returns:
            float: Value of this descriptor for the samples in the dataframe.
        """
        return eval(self.evalstring)  # nosec, pylint: disable=W0123

    def __str__(self):
        """Return string representation of this SISSODescriptor.

        Returns:
            str: String representation of this SISSODescriptor.
        """
        return self.descriptor_string

    @staticmethod
    def _decode_function(string):
        """Get a function based on the string."""
        OPERATORS_REPLACEMENT = [
            "exp(-",
            "exp(",
            "sin(",
            "cos(",
            "sqrt(",
            "cbrt(",
            "log(",
            "abs(",
            "scd(",
            ")^-1",
            ")^2",
            ")^3",
            ")^6",
            "+",
            "-",
            "*",
            "/",
            "(",
            ")",
        ]

        # Get the list of base features needed
        # First replace the operators with "#"
        replaced_string = string
        for op in OPERATORS_REPLACEMENT:
            replaced_string = replaced_string.replace(op, "#" * len(op))
        # Get the features in order of the string and get the unique list of features
        if replaced_string[0] != "#" or replaced_string[-1] != "#":
            raise ValueError('String should start and end with "#"')
        features_in_string = []
        in_feature_word = False
        ichar_start = None
        inputs = []
        for ichar, char in enumerate(replaced_string):
            if in_feature_word and char == "#":
                in_feature_word = False
                featname = replaced_string[ichar_start:ichar]
                if featname not in inputs:
                    inputs.append(featname)
                features_in_string.append(
                    {"featname": featname, "istart": ichar_start, "iend": ichar}
                )
            elif not in_feature_word and char != "#":
                in_feature_word = True
                ichar_start = ichar

        # Prepare string to be formatted from features
        prev_ichar = None
        out = []
        for fdict in features_in_string:
            out.append(string[prev_ichar : fdict["istart"]])
            prev_ichar = fdict["iend"]
            out.append("df['{}']".format(fdict["featname"]))
        out.append(string[prev_ichar:None])
        evalstring = "".join(out)

        # Replace operators in the string with numpy operators
        evalstring = evalstring.replace("sin(", "np.sin(")
        evalstring = evalstring.replace("cos(", "np.cos(")
        evalstring = evalstring.replace("exp(", "np.exp(")
        evalstring = evalstring.replace("log(", "np.log(")
        evalstring = evalstring.replace("sqrt(", "np.sqrt(")
        evalstring = evalstring.replace("cbrt(", "np.cbrt(")
        evalstring = evalstring.replace("abs(", "np.abs(")
        evalstring = evalstring.replace(")^2", ")**2")
        evalstring = evalstring.replace(")^3", ")**3")
        evalstring = evalstring.replace(")^6", ")**6")
        # Deal with the ^-1 ...
        while ")^-1" in evalstring:
            idx1 = evalstring.index(")^-1")
            level = 0
            for ii in range(idx1, -1, -1):
                if evalstring[ii] == ")":
                    level += 1
                elif evalstring[ii] == "(":
                    level -= 1
                if level == 0:
                    idx2 = ii
                    break
            else:
                raise ValueError(r'Could not find initial parenthesis for ")^-1".')
            evalstring = (
                evalstring[:idx2]
                + "1.0/"
                + evalstring[idx2:idx1]
                + ")"
                + evalstring[idx1 + 4 :]
            )

        return {
            "evalstring": evalstring,
            "features_in_string": features_in_string,
            "inputs": inputs,
        }

    @classmethod
    def from_string(cls, string: str):
        """Construct SISSODescriptor from string.

        The string must be the line of the descriptor in the SISSO.out output file,
            e.g. : 1:[((feature1-feature2)+(feature3-feature4))]

        Args:
            string: Substring from the SISSO.out output file corresponding to one
                descriptor of SISSO.
        """
        sp = string.split(":")
        return cls(descriptor_id=int(sp[0]), descriptor_string=sp[1][1:].split("]")[0])


class SISSOModel(MSONable):
    """Class containing one SISSO model."""

    def __init__(
        self,
        dimension: int,
        descriptors: List[SISSODescriptor],
        coefficients: List[List[float]],
        intercept: List[float],
        rmse: Union[List[float], None] = None,
        maxae: Union[List[float], None] = None,
    ):
        """Construct SISSOModel class.

        Args:
            dimension: Dimension of the model.
            descriptors: List of descriptors used in the model.
            coefficients: Coefficient of each descriptor for each task/property.
            intercept: Intercept of the model for each task/property.
            rmse: Root Mean Squared Error of the model on the training data for each
                task/property.
            maxae: Maximum Absolute Error of the model on the training data for each
                task/property.
        """
        self.dimension = dimension
        self.descriptors = descriptors
        self.coefficients = coefficients
        self.intercept = intercept
        self.rmse = rmse
        self.maxae = maxae

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict values from input DataFrame.

        The input DataFrame should have the columns needed by the different SISSO
            descriptors.

        Args:
            df: panda's DataFrame containing the base features needed to apply the
                model.

        Returns:
            darray: Predicted values from the model.
        """
        out = np.array([[intercept] * len(df) for intercept in self.intercept]).T
        for idescr, descr in enumerate(self.descriptors):
            for itask, coeffs in enumerate(self.coefficients):
                dval = coeffs[idescr] * descr.evaluate(df)
                out[:, itask] += dval  # pylint: disable=E1137
        return out

    @classmethod
    def from_string(cls, string: str):
        """Construct SISSOModel object from string.

        The string must be the excerpt corresponding to one model, starting with a
            line of 80 "=" characters and ending with a line of 80 "=" characters.

        Args:
            string: String from the SISSO.out output file corresponding to one model
                of SISSO.
        """
        lines = string.split("\n")
        dimension = int(lines[1].split("D descriptor")[0])
        descriptors: Optional[List[SISSODescriptor]] = None
        coefficients = []
        intercept = []
        rmse = []
        maxae = []
        for _, line in enumerate(lines):
            if "@@@descriptor" in line:
                descriptors = []
                continue
            if descriptors is not None and len(descriptors) < dimension:
                descriptors.append(SISSODescriptor.from_string(line))
                continue
            if "coefficients_" in line:
                coefficients.append([float(nn) for nn in line.split(":")[1].split()])
            elif "Intercept_" in line:
                intercept.append(float(line.split(":")[1]))
            elif "RMSE,MaxAE_" in line:
                sp = line.split()
                rmse.append(float(sp[1]))
                maxae.append(float(sp[2]))

        if descriptors is None:  # pragma: no cover, wrong SISSO output
            raise ValueError("Descriptor not found")

        return cls(
            dimension=dimension,
            descriptors=descriptors,
            coefficients=coefficients,
            intercept=intercept,
            rmse=rmse,
            maxae=maxae,
        )


class SISSOIteration(MSONable):
    """Class containing one SISSO iteration."""

    def __init__(
        self,
        iteration_number: int,
        sisso_model: SISSOModel,
        feature_spaces: Mapping[str, int],
        SIS_subspace_size: int,
        cpu_time: float,
    ):
        """Construct SISSOIteration class.

        Args:
            iteration_number: Number of the iteration.
            sisso_model: SISSO model of this iteration.
            feature_spaces: Number of features in each feature rung.
            SIS_subspace_size: Size of the SIS subspace.
            cpu_time: CPU time for this SISSO iteration.
        """
        self.iteration_number = iteration_number
        self.sisso_model = sisso_model
        self.feature_spaces = feature_spaces
        self.SIS_subspace_size = SIS_subspace_size
        self.cpu_time = cpu_time

    @classmethod
    def from_string(cls, string: str):
        """Construct SISSOIteration object from string.

        The string must be the excerpt corresponding to one iteration, i.e. it must
        start with "iteration:   N" and end with "DI done!".

        Args:
            string: String from the SISSO.out output file corresponding to one
                iteration of SISSO.
        """
        lines = string.split("\n")
        it_num = int(lines[0].split(":")[1])

        r_sisso_model = r"={80}.*?={80}"
        match_sisso_model = re.findall(r_sisso_model, string, re.DOTALL)
        if len(match_sisso_model) != 1:  # pragma: no cover, wrong SISSO output
            raise ValueError(
                "Should get exactly one SISSO model excerpt in the string."
            )
        sisso_model = SISSOModel.from_string(match_sisso_model[0])

        # TODO: different for iteration 1 !
        r_feature_spaces = r"Total number of features in the space phi.*?\n"
        match_feature_spaces = re.findall(r_feature_spaces, string)
        feature_spaces = {
            mfs.split()[-2][:-1]: int(mfs.split()[-1]) for mfs in match_feature_spaces
        }

        r_SIS_subspace_size = r"Size of the SIS-selected subspace.*?\n"
        match_SIS_subspace_size = re.findall(r_SIS_subspace_size, string)
        if len(match_SIS_subspace_size) != 1:  # pragma: no cover, wrong SISSO output
            raise ValueError("Should get exactly one SIS subspace size in the string.")
        SIS_subspace_size = int(match_SIS_subspace_size[0].split()[-1])

        r_cputime = r"Wall-clock time \(second\) for this DI:.*?\n"
        match_cputime = re.findall(r_cputime, string)
        if len(match_cputime) != 1:  # pragma: no cover, wrong SISSO output
            raise ValueError(
                "Should get exactly one Wall-clock time in the string, "
                "got {:d}.".format(len(match_cputime))
            )
        cpu_time = float(match_cputime[0].split()[-1])

        return cls(
            iteration_number=it_num,
            sisso_model=sisso_model,
            feature_spaces=feature_spaces,
            SIS_subspace_size=SIS_subspace_size,
            cpu_time=cpu_time,
        )


class SISSOParams(MSONable):
    """Class containing input parameters extracted from the SISSO output file."""

    PARAMS: List[Tuple[str, str, Union[type, Callable]]] = [
        ("property_type", "Descriptor dimension:", int),
        ("descriptor_dimension", "Descriptor dimension:", int),
        ("total_number_properties", "Total number of properties:", int),
        ("task_weighting", "Task_weighting:", list_of_ints),
        ("number_of_samples", "Number of samples for each property:", list_of_ints),
        ("n_scalar_features", "Number of scalar features:", int),
        (
            "n_rungs",
            r"Number of recursive calls for feature transformation \(rung of the "
            r"feature space\):",
            int,
        ),
        (
            "max_feature_complexity",
            r"Max feature complexity \(number of operators in a feature\):",
            int,
        ),
        (
            "n_dimension_types",
            r"Number of dimension\(unit\)-type \(for dimension analysis\):",
            int,
        ),
        (
            "dimension_types",
            "Dimension type for each primary feature:",
            matrix_of_floats,
        ),
        (
            "lower_bound_maxabs_value",
            r"Lower bound of the max abs\. data value for the selected features:",
            float,
        ),
        (
            "upper_bound_maxabs_value",
            r"Upper bound of the max abs\. data value for the selected features:",
            float,
        ),
        (
            "SIS_subspaces_sizes",
            r"Size of the SIS-selected \(single\) subspace :",
            list_of_ints,
        ),
        ("operators", "Operators for feature construction:", list_of_strs),
        ("sparsification_method", "Method for sparsification:", str),
        ("n_topmodels", "Number of the top ranked models to output:", int),
        ("fit_intercept", r"Fitting intercept\?", str_to_bool),
        ("metric", "Metric for model selection:", str),
    ]

    def __init__(
        self,
        property_type: int,
        descriptor_dimension: int,
        total_number_properties: int,
        task_weighting: List[int],
        number_of_samples: List[int],
        n_scalar_features: int,
        n_rungs: int,
        max_feature_complexity: int,
        n_dimension_types: int,
        dimension_types: int,
        lower_bound_maxabs_value: float,
        upper_bound_maxabs_value: float,
        SIS_subspaces_sizes: List[int],
        operators: List[str],
        sparsification_method: str,
        n_topmodels: int,
        fit_intercept: bool,
        metric: str,
    ):  # noqa: D417
        """Construct SISSOParams class.

        All arguments not listed below are arguments from the SISSO code. For more
        information, see https://github.com/rouyang2017/SISSO.
        """
        self.property_type = property_type
        self.descriptor_dimension = descriptor_dimension
        self.total_number_properties = total_number_properties
        self.task_weighting = task_weighting
        self.number_of_samples = number_of_samples
        self.n_scalar_features = n_scalar_features
        self.n_rungs = n_rungs
        self.max_feature_complexity = max_feature_complexity
        self.n_dimension_types = n_dimension_types
        self.dimension_types = dimension_types
        self.lower_bound_maxabs_value = lower_bound_maxabs_value
        self.upper_bound_maxabs_value = upper_bound_maxabs_value
        self.SIS_subspaces_sizes = SIS_subspaces_sizes
        self.operators = operators
        self.sparsification_method = sparsification_method
        self.n_topmodels = n_topmodels
        self.fit_intercept = fit_intercept
        self.metric = metric

    @classmethod
    def from_string(cls, string: str):
        """Construct SISSOParams object from string."""
        kwargs = {}
        for class_var, output_var_str, var_type in cls.PARAMS:
            if class_var == "dimension_types":
                match = re.search(
                    r"{}(.*?)\nLower bound".format(output_var_str), string, re.DOTALL
                )
                if match is None:  # pragma: no cover, wrong SISSO output
                    raise ValueError(
                        'Should find a match for "{}"'.format(output_var_str)
                    )
                kwargs[class_var] = var_type(match.group(1).strip())
            else:
                matches = re.findall(r"{}.*?\n".format(output_var_str), string)
                if len(matches) != 1:  # pragma: no cover, wrong SISSO output
                    raise ValueError(
                        'Should get exactly one match for "{}".'.format(output_var_str)
                    )
                kwargs[class_var] = var_type(matches[0].split()[-1])
        return cls(**kwargs)

    def __str__(self):
        """Return string representation of the SISSO parameters.

        Returns:
            str: String representation of this SISSOParams object.
        """
        out = ["Parameters for SISSO :"]
        for class_var, _, _ in self.PARAMS:
            out.append(
                " - {} : {}".format(class_var, str(self.__getattribute__(class_var)))
            )
        return "\n".join(out)


class SISSOOut(MSONable):
    """Class containing the results contained in the SISSO output file (SISSO.out)."""

    def __init__(
        self,
        params: SISSOParams,
        iterations: List[SISSOIteration],
        version: SISSOVersion,
        cpu_time: Optional[float],
    ):
        """Construct SISSOOut class.

        Args:
            params: Parameters used for SISSO (as a SISSOParams object).
            iterations: List of SISSO iterations.
            version: Information about the version of SISSO used as a SISSOVersion
                object.
            cpu_time: Wall-clock CPU time from the output file.
        """
        self.params = params
        self.iterations = iterations
        self.version = version
        self.cpu_time = cpu_time

    @classmethod
    def from_file(cls, filepath: str = "SISSO.out", allow_unfinished: bool = False):
        """Read in SISSOOut data from file.

        Args:
            filepath: Path of the file to extract output.
            allow_unfinished: Whether to allow parsing of unfinished SISSO runs.
        """
        with open(filepath, "r") as f:
            string = f.read()

        r = r"Reading parameters from SISSO\.in:\s?\n-{80}.*?-{80}"
        match = re.findall(r, string, re.DOTALL)
        if len(match) != 1:  # pragma: no cover, wrong SISSO output
            raise ValueError(
                "Should get exactly one excerpt for input parameters in the string."
            )
        params = SISSOParams.from_string(match[0])

        r = r"iteration:.*?DI done!"
        match = re.findall(r, string, re.DOTALL)

        iterations = []
        for iteration_string in match:
            iterations.append(SISSOIteration.from_string(iteration_string))

        r = r"Total wall-clock time \(second\):.*?\n"
        match = re.findall(r, string)
        if len(match) == 0:
            if not allow_unfinished:
                raise ValueError(
                    "Should get exactly one total cpu time in the string, got 0."
                )
            cpu_time = None
        elif len(match) > 1:  # pragma: no cover, wrong SISSO output
            raise ValueError(
                "Should get exactly one total cpu time in the string, "
                "got {:d}.".format(len(match))
            )
        else:
            cpu_time = float(match[0].split()[-1])

        with open(filepath, "r") as f:
            header = f.readline()
        version = SISSOVersion.from_string(header)

        return cls(
            params=params, iterations=iterations, version=version, cpu_time=cpu_time
        )

    @property
    def model(self):
        """Model for this SISSO run.

        The last model is provided (with the highest dimension).
        """
        return self.iterations[-1].sisso_model

    @property
    def models(self):
        """Models (for all dimensions) for this SISSO run."""
        return [it.sisso_model for it in self.iterations]


class TopModels(MSONable):
    """Class containing summary info of the top N models from SISSO.

    This class is a container for the topNNNN_DDDd files (NNNN being the number of
    models in the file and DDD the dimension of the descriptor) that are stored in
    the models directory.
    """


class TopModelsCoefficients(MSONable):
    """Class containing the coefficients of the features for the top N models.

    This class is a container for the topNNNN_DDDd_coeff files (NNNN being the number
    of models in the file and DDD the dimension of the descriptor) that are stored
    in the models directory.
    """


class FeatureSpace(MSONable):
    """Class containing the selected features from SISSO.

    This class is a container for the space_DDDd.name files (DDD being the dimension
    of the descriptor) that are stored in the feature_space directory.
    """


class DescriptorsDataModels(MSONable):  # pragma: no cover, reading full SISSO.out
    """Class containing the true and predicted data for the best descriptors/models.

    This class is a container for the desc_DDDd_pPPP.dat files (DDD being the
    dimension of the descriptor and PPP the property number in case of multi-task
    SISSO) that are stored in the desc_dat directory.

    Note: see if we want to implement this class, everything might be contained in
        SISSO.out and its SISSOOut object.
    """

    def __init__(self, data):
        """Construct this DescriptorsDataModels object.

        Args:
            data: Data for this DescriptorsDataModels object.
        """
        self.data = data

    @classmethod
    def from_file(cls, filepath):
        """Construct this DescriptorsDataModels object from file.

        Args:
            filepath: File to construct this DescriptorsDataModels from.

        Returns:
            DescriptorsDataModels: DescriptorsDataModels object.
        """
        if filepath.endswith(".dat"):
            return cls.from_dat_file(filepath)
        else:
            raise ValueError("The from_file method is working only with .dat files")

    @classmethod
    def from_dat_file(cls, filepath):
        """Construct this DescriptorsDataModels object from .dat file.

        Args:
            filepath: File to construct this DescriptorsDataModels from.

        Returns:
            DescriptorsDataModels: DescriptorsDataModels object.
        """
        data = pd.read_csv(filepath, delim_whitespace=True)
        return cls(data=data)


class ResidualData(MSONable):
    """Class containing the residuals for the training data computed at each iteration.

    This class is a container for the res_DDDd_pPPP.dat files (DDD being the dimension
    of the descriptor and PPP the property number in case of multi-task SISSO)
    that are stored in the residual directory.
    """
