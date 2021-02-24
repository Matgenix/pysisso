# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL, All rights reserved.
# Distributed open source for academic and non-profit users.
# Contact Matgenix for commercial usage.
# See LICENSE file for details.

"""Module containing classes to create and manipulate SISSO input files."""

import datetime
from typing import List, Union

import pandas as pd  # type: ignore
from monty.json import MSONable  # type: ignore


class SISSODat(MSONable):
    """Main class containing the data for SISSO (training, test or new data)."""

    def __init__(
        self,
        data: pd.DataFrame,
        features_dimensions: Union[dict, None] = None,
        model_type: str = "regression",
        nsample: Union[List[int], int, None] = None,
    ):
        """Construct SISSODat class.

        The input data must be a pandas DataFrame for which the first column contains
        the identifiers for each data point (e.g. material identifier, batch number of
        a process, ...), the second column contains the property to be predicted and
        the other columns are the base features.

        Classification is not yet supported (needs the items in the same classes to
        be grouped together).

        Args:
            data: Input data as pandas DataFrame object. The first column must be the
                identifiers for each data point, the second column must be the property
                to be predicted, and the other columns are the base features.
            features_dimensions: Dimension of the different base features as a
                dictionary mapping the name of each feature to its dimension.
                Features not in the dictionary are supposed to be dimensionless.
                If set to None, all features are supposed to be dimensionless.
            model_type: Type of model. Should be either "regression" or
                "classification".
            nsample: Number of samples. If None or an integer, SISSO is supposed to be
                Single-Task (ST). If a list of integers, SISSO is supposed to be
                Multi-Task (MT).

        Raises:
            ValueError: if nsample is not compatible with the data frame.

        Notes:
            The pandas index has not be used for the identifier here. Indeed when
            Multi-Task SISSO is used, the same identifier can occur for two different
            tasks/properties.
        """
        self.data = data
        self.features_dimensions = features_dimensions
        self.model_type = model_type
        self.nsample = nsample
        self._order_features()

    def _order_features(self):
        if self.features_dimensions is None:
            return
        if len(self.features_dimensions) == 0:
            return
        if "_NODIM" in self.features_dimensions.values():
            raise ValueError(
                'Dimension name "_NODIM" in features_dimensions is not allowed.'
            )
        cols = list(self.data.columns)
        if self.model_type == "regression":
            ii = 2
        elif self.model_type == "classification":  # pragma: no cover
            ii = 1
        else:  # pragma: no cover # should not be anything else
            raise ValueError("Wrong model_type")
        newcols = cols[:ii]
        featcols = cols[ii:]
        newcols.extend(
            sorted(
                featcols,
                key=lambda x: self.features_dimensions[x]
                if x in self.features_dimensions
                else "_NODIM",
            )
        )
        self.data = self.data[newcols]

    @property
    def SISSO_features_dimensions_ranges(self):
        """Get the ranges of features for each dimension.

        Returns:
            dict: Dimension to range mapping.
        """
        cols = list(self.data.columns)
        if self.model_type == "regression":
            ii = 2
        elif self.model_type == "classification":  # pragma: no cover
            ii = 1
        else:  # pragma: no cover # should not be anything else
            raise ValueError("Wrong model_type")
        featcols = cols[ii:]
        featdimensions = [
            self.features_dimensions[featcol]
            if featcol in self.features_dimensions
            else None
            for featcol in featcols
        ]
        uniquedimensions = list(set(featdimensions))
        ranges = {}
        for dimension in uniquedimensions:
            idx = featdimensions.index(dimension)
            count = featdimensions.count(dimension)
            ranges[dimension] = (idx + 1, idx + count)
        # Check that the ranges do not overlap
        for dim1, range1 in ranges.items():
            for dim2, range2 in ranges.items():
                if dim1 == dim2:
                    continue
                if self._check_ranges_overlap(range1, range2):  # pragma: no cover
                    raise ValueError("Dimension ranges overlap :")
        return ranges

    @staticmethod
    def _check_ranges_overlap(r1, r2):
        return not (
            (r1[0] < r2[0] and r1[1] < r2[0]) or (r2[0] < r1[0] and r2[1] < r1[0])
        )

    @property
    def nsample(self):
        """Return number of samples in this data set.

        Returns:
            int: Number of samples
        """
        return self._nsample

    @nsample.setter
    def nsample(self, nsample):
        if nsample is None:
            self._nsample = len(self.data)
        elif isinstance(nsample, int):
            if nsample != len(self.data):
                raise ValueError("The size of the DataFrame does not match nsample.")
            self._nsample = nsample
        elif isinstance(nsample, list):
            if sum(nsample) != len(self.data):  # pragma: no cover
                raise ValueError(
                    "Sum of all samples is not equal to the size of the DataFrame."
                )
            self._nsample = nsample
        else:
            raise ValueError(
                'Type "{}" is not valid for nsample.'.format(type(nsample))
            )

    @property
    def ntask(self):
        """Return number of tasks (i.e. output targets) in this data set.

        Returns:
            int: Number of tasks
        """
        if isinstance(self.nsample, int):
            return 1
        elif isinstance(self.nsample, list):
            return len(self.nsample)
        else:  # pragma: no cover
            raise ValueError("Wrong nsample in SISSODat.")

    @property
    def nsf(self):
        """Return number of (scalar) features in this data set.

        Returns:
            int: Number of (scalar) features.
        """
        return len(self.data.columns) - 2

    @property
    def input_string(self):
        """Input string of the .dat file.

        Returns:
            str: String for the .dat file.
        """
        out = [
            " ".join(["{:20}".format(column_name) for column_name in self.data.columns])
        ]
        max_str_size = max(self.data[self.data.columns[0]].apply(len))
        header_row_format_str = "{{:{}}}".format(max(20, max_str_size))
        for _, row in self.data.iterrows():
            row_list = list(row)
            line = [header_row_format_str.format(row_list[0])]
            # line = ['{:20}'.format(row_list[0])]
            for col in row_list[1:]:
                line.append("{:<20.12f}".format(col))
            out.append(" ".join(line))
        return "\n".join(out)

    def to_file(self, filename="train.dat"):
        """Write this SISSODat object to file.

        Args:
            filename: Name of the file to write this SISSODat to.
        """
        with open(filename, "w") as f:
            f.write(self.input_string)

    @classmethod
    def from_file(cls, filepath, features_dimensions=None):
        """Construct SISSODat object from file.

        Args:
            filepath: Name of the file.
            features_dimensions: Dimension of the different base features as a
                dictionary mapping the name of each feature to its dimension.
                Features not in the dictionary are supposed to be dimensionless.
                If set to None, all features are supposed to be dimensionless.

        Returns:
            SISSODat: SISSODat object extracted from file.
        """
        if filepath.endswith(".dat"):
            return cls.from_dat_file(filepath, features_dimensions=features_dimensions)
        else:  # pragma: no cover
            raise ValueError("The from_file method is working only with .dat files")

    @classmethod
    def from_dat_file(cls, filepath, features_dimensions=None, nsample=None):
        """Construct SISSODat object from .dat file.

        Args:
            filepath: Name of the file.
            features_dimensions: Dimension of the different base features as a
                dictionary mapping the name of each feature to its dimension.
                Features not in the dictionary are supposed to be dimensionless.
                If set to None, all features are supposed to be dimensionless.
            nsample: Number of samples in the .dat file. If set to None, will be set
                automatically.

        Returns:
            SISSODat: SISSODat object extracted from file.
        """
        data = pd.read_csv(filepath, delim_whitespace=True)
        return cls(data=data, features_dimensions=features_dimensions, nsample=nsample)


class SISSOIn(MSONable):
    """Main class containing the input variables for SISSO.

    This class is basically a container for the SISSO.in input file for SISSO.
    Additional helper functions are available.
    """

    #: dict: Types or descriptions (as a string) of the values for each SISSO keyword.
    KW_TYPES = {
        "ptype": tuple([int]),
        "ntask": tuple([int]),
        "nsample": tuple([int, "list_of_ints"]),
        "task_weighting": tuple([int]),
        "desc_dim": tuple([int]),
        "nsf": tuple([int]),
        "restart": tuple([bool]),
        "rung": tuple([int]),
        "opset": tuple(["str_operators"]),
        "maxcomplexity": tuple([int]),
        "dimclass": tuple(["str_dimensions"]),
        "maxfval_lb": tuple([float]),
        "maxfval_ub": tuple([float]),
        "subs_sis": tuple([int, "list_of_ints"]),
        "method": tuple([str]),
        "L1L0_size4L0": tuple([int]),
        "fit_intercept": tuple([bool]),
        "metric": tuple([str]),
        "nm_output": tuple([int]),
        "isconvex": tuple(["str_isconvex"]),
        "width": tuple([float]),
        "nvf": tuple([int]),
        "vfsize": tuple([int]),
        "vf2sf": tuple([str]),
        "npf_must": tuple([int]),
        "L1_max_iter": tuple([int]),
        "L1_tole": tuple([float]),
        "L1_dens": tuple([int]),
        "L1_nlambda": tuple([int]),
        "L1_minrmse": tuple([float]),
        "L1_warm_start": tuple([bool]),
        "L1_weighted": tuple([bool]),
    }

    #: dict: Available unary and binary operators for feature construction.
    # TODO: add string description
    AVAILABLE_OPERATIONS = {
        "unary": {
            "exp": "",
            "exp-": "",
            "^-1": "",
            "scd": "",
            "^2": "",
            "^3": "",
            "^6": "",
            "sqrt": "",
            "cbrt": "",
            "log": "",
            "sin": "",
            "cos": "",
        },
        "binary": {"+": "", "-": "", "|-|": "", "*": "", "/": ""},
    }

    def __init__(
        self,
        target_properties_keywords,
        feature_construction_sure_independence_screening_keywords,
        descriptor_identification_keywords,
        fix=False,
    ):
        """Construct SISSOIn object.

        Args:
            target_properties_keywords: Keywords related to target properties.
            feature_construction_sure_independence_screening_keywords: Keywords related
                to feature construction and sure independence screening.
            descriptor_identification_keywords: Keywords related to descriptor
                identification.
            fix: Whether to automatically fix keywords when they are not compatible.
        """
        self.target_properties_keywords = target_properties_keywords
        self.feature_construction_sure_independence_screening_keywords = (
            feature_construction_sure_independence_screening_keywords
        )
        self.descriptor_identification_keywords = descriptor_identification_keywords
        self._check_keywords(fix=fix)

    def _check_keywords(self, fix=False):
        # TODO: implement a check on the keywords
        # When using L1L0 method, L1L0_size4L0 should not be > subs_sis,
        #   i.e. should be <= subs_sis ("STOP Error: fs_size_L0 must not larger than
        #       fs_size_DI !" in SISSO.err)
        # When using L1L0 method, L1L0_size4L0 should be >= desc_dim
        #   i.e. it crashes when it reaches a dimension larger than L1L0_size4L0
        #   ("Program received signal SIGSEGV: Segmentation fault - invalid memory
        #       reference." in SISSO.err)
        # * L1L0_size4L0 <= subs_sis
        # * L1L0_size4L0 >= desc_dim
        # In short :
        # desc_dim <= L1L0_size4L0 <= subs_sis
        # Possible fixes :
        # A. When the number of features is large, fix L1L0_size4L0 and subs_sis:
        #   A.1. increase L1L0_size4L0 to at least desc_dim
        #   A.2. increase subs_sis to at least L1L0_size4L0
        # B. When the number of features is small, we get the following message
        #   in SISSO.log :
        #   "# WARNING: the actual size of the selected subspace is smaller than that
        #       specified in "SISSO.in" !!!"
        #   In that case, subs_sis cannot be increased, L1L0_size4L0 has to be
        #       decreased, and in any case, the number of descriptors (desc_dim)
        #       cannot be larger than L1L0_size4L0
        # In all method cases (L0 or L1L0), when desc_dim is larger than the total
        #   number of features, we get :
        #   "Program received signal SIGSEGV: Segmentation fault - invalid memory
        #       reference." in SISSO.err
        uses_L1L0 = self.descriptor_identification_keywords["method"] == "L1L0"
        if uses_L1L0:
            desc_dim = self.target_properties_keywords["desc_dim"]
            L1L0_size4L0 = self.descriptor_identification_keywords["L1L0_size4L0"]
            subs_sis = self.feature_construction_sure_independence_screening_keywords[
                "subs_sis"
            ]
            if desc_dim > L1L0_size4L0:
                if not fix:
                    raise ValueError(
                        "Dimension of descriptor (desc_dim={:d}) is larger than the "
                        "number of features available for L0 norm from L1 screening "
                        "(L1L0_size4L0={:d}).".format(desc_dim, L1L0_size4L0)
                    )
                L1L0_size4L0 = desc_dim
                self.descriptor_identification_keywords["L1L0_size4L0"] = L1L0_size4L0
            if isinstance(subs_sis, int):
                if L1L0_size4L0 > subs_sis:
                    if not fix:
                        raise ValueError(
                            "Number of features to be screened by L1 for L0 "
                            "(L1L0_size4L0={:d}) is larger than SIS-selected subspace "
                            "(subs_sis={:d}).".format(L1L0_size4L0, subs_sis)
                        )
                    self.feature_construction_sure_independence_screening_keywords[
                        "subs_sis"
                    ] = L1L0_size4L0
            elif isinstance(subs_sis, list):
                subs_sis_list = list(subs_sis)
                for dim, subs_sis_dim in enumerate(subs_sis_list, start=1):
                    if L1L0_size4L0 > subs_sis_dim:
                        if not fix:
                            raise ValueError(
                                "Number of features to be screened by L1 for L0 "
                                "(L1L0_size4L0={:d}) is larger than SIS-selected "
                                "subspace (subs_sis={:d}) of dimension {:d}.".format(
                                    L1L0_size4L0, subs_sis_dim, dim
                                )
                            )
                        subs_sis[dim - 1] = L1L0_size4L0
            else:  # pragma: no cover, should never be here after kw formats check
                raise ValueError(
                    'Wrong type for "subs_sis" : {}'.format(type(subs_sis))
                )

    def _format_kw_value(self, kw, val, float_format=".12f"):
        allowed_types = self.KW_TYPES[kw]
        # Determine the type of the value for this keyword
        val_type = None
        for allowed_type in allowed_types:
            if allowed_type is int:
                if type(val) is int:
                    val_type = int
                    break
            elif allowed_type is float:
                if type(val) is float:  # pragma: no branch
                    val_type = float
                    break
            elif allowed_type is bool:
                if type(val) is bool:  # pragma: no branch
                    val_type = bool
                    break
            elif allowed_type is str:
                if type(val) is str:  # pragma: no branch
                    val_type = str
                    break
            elif allowed_type == "list_of_ints":
                if (  # pragma: no branch
                    type(val) is list or type(val) is tuple
                ) and all([type(item) is int for item in val]):
                    val_type = "list_of_ints"
                    break
            # TODO: add checks on the str_operators, str_dimensions and str_isconvex
            elif allowed_type == "str_operators":
                val_type = "str_operators"
            elif allowed_type == "str_dimensions":  # pragma: no cover
                val_type = "str_dimensions"
            elif allowed_type == "str_isconvex":  # pragma: no cover
                val_type = "str_isconvex"
        if val_type is None:  # pragma: no cover
            raise ValueError(
                'Type of value "{}" for keyword "{}" not found/valid.'.format(
                    str(val), kw
                )
            )

        if val_type is int:
            return "{}={:d}".format(kw, val)
        elif val_type is float:
            float_ref_str = "{}={{:{}}}".format(kw, float_format)
            return float_ref_str.format(val)
        elif val_type is bool:
            return "{}=.{}.".format(kw, str(val).lower())
        elif val_type is str:
            return "{}='{}'".format(kw, val)
        elif val_type == "list_of_ints":
            if kw in ["subs_sis", "nsample"]:
                return "{}={}".format(kw, ",".join(["{:d}".format(v) for v in val]))
            else:  # pragma: no cover
                return "{}=({})".format(kw, ",".join(["{:d}".format(v) for v in val]))
        elif val_type == "str_operators":
            return "{}='{}'".format(kw, val)
        elif val_type in ["str_dimensions", "str_isconvex"]:  # pragma: no cover
            return "{}={}".format(kw, val)
        else:  # pragma: no cover
            raise ValueError(
                "Wrong type for SISSO value.\nSISSO keyword : {}\n"
                "Value : {} (type : {})".format(kw, str(val), val_type)
            )

    def input_string(self, matgenix_acknowledgement=True):
        """Input string of the SISSO.in file.

        Args:
            matgenix_acknowledgement: Whether to add the acknowledgment of Matgenix.

        Returns:
            str: String for the SISSO.in file.
        """
        if (
            self.target_properties_keywords["nsample"] is None
            or self.feature_construction_sure_independence_screening_keywords["nsf"]
            is None
        ):  # pragma: no cover # unlikely wrong usage
            raise ValueError(
                'Both keywords "nsample" and "nsf" should be set to get SISSO.in\'s '
                "input_string"
            )
        out = []
        if matgenix_acknowledgement:
            year = datetime.datetime.now().year
            out.append(
                "!------------------------------------------------------------!\n"
                "! SISSO.in generated by Matgenix's pysisso package.          !\n"
                "! Copyright (c) {:d}, Matgenix SRL. All Rights Reserved.     !\n"
                "! Distributed open source for academic and non-profit users. !\n"
                "! Contact Matgenix for commercial usage.                     !\n"
                "! See LICENSE file for details.                              !\n"
                "!------------------------------------------------------------!"
                "\n".format(year)
            )
        if self.is_regression:
            out.append(
                "!------------------!\n"
                "! REGRESSION MODEL !\n"
                "!------------------!\n"
            )
        elif self.is_classification:  # pragma: no cover # not yet implemented
            out.append(
                "!----------------------!\n"
                "! CLASSIFICATION MODEL !\n"
                "!----------------------!\n"
            )

        # Keywords related to target properties
        out.append(
            "!------------------------------------!\n"
            "! Keywords for the target properties !\n"
            "!------------------------------------!"
        )
        for sisso_kw, sisso_val in self.target_properties_keywords.items():
            if sisso_val is None:
                continue
            out.append(self._format_kw_value(kw=sisso_kw, val=sisso_val))
        out.append("")

        # Keywords related to feature construction (FC) and
        #  sure independence screening (SIS)
        out.append(
            "!----------------------------------------"
            "--------------------------------------!\n"
            "! Keywords for feature construction (FC) "
            "and sure independence screening (SIS) !\n"
            "!----------------------------------------"
            "--------------------------------------!"
        )
        for (
            sisso_kw,
            sisso_val,
        ) in self.feature_construction_sure_independence_screening_keywords.items():
            if sisso_val is None:
                continue
            out.append(self._format_kw_value(kw=sisso_kw, val=sisso_val))
        out.append("")

        # Keywords descriptor identification via a sparsifying operator
        out.append(
            "!------------------------------------------------------------------!\n"
            "! Keyword for descriptor identification via a sparsifying operator !\n"
            "!------------------------------------------------------------------!"
        )
        for sisso_kw, sisso_val in self.descriptor_identification_keywords.items():
            if sisso_val is None:
                continue
            out.append(self._format_kw_value(kw=sisso_kw, val=sisso_val))
        return "\n".join(out)

    @property
    def is_regression(self) -> bool:
        """Whether this SISSOIn object corresponds to a regression model.

        Returns:
            bool: True if this SISSOIn object is a regression model, False otherwise.
        """
        return self.target_properties_keywords["ptype"] == 1

    @property
    def is_classification(self):
        """Whether this SISSOIn object corresponds to a classification model.

        Returns:
            bool: True if this SISSOIn object is a classification model,
                False otherwise.
        """
        return self.target_properties_keywords["ptype"] == 2

    @classmethod
    def from_sisso_keywords(
        cls,
        ptype,
        nsample=None,
        nsf=None,
        ntask=1,
        task_weighting=1,
        desc_dim=2,
        restart=False,
        rung=2,
        opset="(+)(-)",
        maxcomplexity=10,
        dimclass=None,
        maxfval_lb=1e-3,
        maxfval_ub=1e5,
        subs_sis=20,
        method="L0",
        L1L0_size4L0=1,
        fit_intercept=True,
        metric="RMSE",
        nm_output=100,
        isconvex=None,
        width=None,
        nvf=None,
        vfsize=None,
        vf2sf=None,
        npf_must=None,
        L1_max_iter=None,
        L1_tole=None,
        L1_dens=None,
        L1_nlambda=None,
        L1_minrmse=None,
        L1_warm_start=None,
        L1_weighted=None,
        fix=False,
    ):  # noqa: D417
        """Construct SISSOIn object from SISSO input keywords.

        Args:
            fix: Whether to fix keywords if they are not compatible.

        Returns:
            SISSOIn: SISSOIn object containing the SISSO input arguments.
        """
        tp_kwds = dict()
        tp_kwds["ptype"] = ptype
        tp_kwds["ntask"] = ntask
        tp_kwds["nsample"] = nsample
        tp_kwds["task_weighting"] = task_weighting
        tp_kwds["desc_dim"] = desc_dim
        tp_kwds["restart"] = restart
        fcsis_kwds = dict()
        fcsis_kwds["nsf"] = nsf
        fcsis_kwds["rung"] = rung
        fcsis_kwds["opset"] = opset
        fcsis_kwds["maxcomplexity"] = maxcomplexity
        fcsis_kwds["dimclass"] = dimclass
        fcsis_kwds["maxfval_lb"] = maxfval_lb
        fcsis_kwds["maxfval_ub"] = maxfval_ub
        fcsis_kwds["subs_sis"] = subs_sis
        fcsis_kwds["nvf"] = nvf
        fcsis_kwds["vfsize"] = vfsize
        fcsis_kwds["vf2sf"] = vf2sf
        fcsis_kwds["npf_must"] = npf_must
        di_kwds = dict()
        di_kwds["method"] = method
        di_kwds["L1L0_size4L0"] = L1L0_size4L0
        di_kwds["fit_intercept"] = fit_intercept
        di_kwds["metric"] = metric
        di_kwds["nm_output"] = nm_output
        di_kwds["isconvex"] = isconvex
        di_kwds["width"] = width
        di_kwds["L1_max_iter"] = L1_max_iter
        di_kwds["L1_tole"] = L1_tole
        di_kwds["L1_dens"] = L1_dens
        di_kwds["L1_nlambda"] = L1_nlambda
        di_kwds["L1_minrmse"] = L1_minrmse
        di_kwds["L1_warm_start"] = L1_warm_start
        di_kwds["L1_weighted"] = L1_weighted
        return cls(
            target_properties_keywords=tp_kwds,
            feature_construction_sure_independence_screening_keywords=fcsis_kwds,
            descriptor_identification_keywords=di_kwds,
            fix=fix,
        )

    @classmethod
    def from_file(cls, filepath):
        """Construct SISSOIn from file.

        Args:
            filepath: Path of the file.
        """
        raise NotImplementedError

    def to_file(
        self,
        filename="SISSO.in",
    ):
        """Write SISSOIn object to file.

        Args:
            filename: Name of the file to write SISSOIn object.
        """
        with open(filename, "w") as f:
            f.write(self.input_string())

    def set_keywords_for_SISSO_dat(self, sisso_dat):
        """Update keywords for a given SISSO dat object.

        Args:
            sisso_dat: SISSODat object to update related keywords.
        """
        dimclass = None
        if sisso_dat.features_dimensions is not None:
            feature_dimensions_ranges = sisso_dat.SISSO_features_dimensions_ranges
            if len(feature_dimensions_ranges) == 0 or (
                len(feature_dimensions_ranges) == 1
                and list(feature_dimensions_ranges.keys())[0] is None
            ):
                dimclass = None
            else:
                dimclasslist = []
                for dim, dimrange in feature_dimensions_ranges.items():
                    if dim is None:
                        continue
                    dimclasslist.append("({:d}:{:d})".format(dimrange[0], dimrange[1]))
                dimclass = "".join(dimclasslist)
        self.target_properties_keywords["nsample"] = sisso_dat.nsample
        self.target_properties_keywords["ntask"] = sisso_dat.ntask
        self.feature_construction_sure_independence_screening_keywords[
            "nsf"
        ] = sisso_dat.nsf
        self.feature_construction_sure_independence_screening_keywords[
            "dimclass"
        ] = dimclass

    @classmethod
    def from_SISSO_dat(
        cls, sisso_dat: SISSODat, model_type: str = "regression", **kwargs: object
    ):
        """Construct SISSOIn object from SISSODat object.

        Args:
            sisso_dat: SISSODat object containing the data to fit.
            model_type: Type of model. Should be "regression" or "classification".
            **kwargs: Keywords to be passed to SISSOIn.

        Returns:
            SISSOIn: SISSOIn object containing all the relevant SISSO input keywords.
        """
        if model_type == "regression":
            ptype = 1
        elif model_type == "classification":
            raise NotImplementedError
        else:
            raise ValueError(
                'Wrong model_type ("{}"). Should be "regression" or '
                '"classification".'.format(model_type)
            )
        sissoin = cls.from_sisso_keywords(ptype=ptype, **kwargs)
        sissoin.set_keywords_for_SISSO_dat(sisso_dat=sisso_dat)
        return sissoin
