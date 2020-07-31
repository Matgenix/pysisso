# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


from monty.json import MSONable
import datetime
import pandas as pd
import numpy as np
from typing import Union


class SISSODat(MSONable):
    """Main class containing the data for SISSO (training data, test data or new data).
    """

    def __init__(self, data: pd.DataFrame, features_dimensions: Union[dict, None]=None,
                 model_type: str = 'regression'):
        """Constructor for SISSODat class.

        The input data must be a pandas DataFrame for which the first column contains
        the identifiers for each data point (e.g. material identifier, batch number of a process, ...),
        the second column contains the property to be predicted and the other columns are the base features.

        Classification is not yet supported (needs the items in the same classes to be grouped together).
        Multi-Task SISSO is not yet supported.

        Args:
            data: Input data as pandas DataFrame object. The first column must be the identifiers for each data point,
                the second column must be the property to be predicted, and the other columns are the base features.
            features_dimensions: Dimension of the different base features as a dictionary mapping the name of each
                feature to its dimension. Features not in the dictionary are supposed to be dimensionless. If set to
                None, all features are supposed to be dimensionless.
            model_type: Type of model. Should be either "regression" or "classification".
        """
        self.data = data
        self.features_dimensions = features_dimensions
        self.model_type = model_type
        self._order_features()

    def _order_features(self):
        if self.features_dimensions is None:
            return
        if len(self.features_dimensions) == 0:
            return
        if '_NODIM' in self.features_dimensions:
            raise ValueError('Dimension name "_NODIM" in features_dimensions is not allowed.')
        cols = list(self.data.columns)
        if self.model_type == 'regression':
            ii = 2
        elif self.model_type == 'classification':
            ii = 1
        else:
            raise ValueError('Wrong model_type')
        newcols = cols[:ii]
        featcols = cols[ii:]
        newcols.extend(sorted(featcols,
                              key=lambda x: self.features_dimensions[x] if x in self.features_dimensions else '_NODIM'))
        self.data = self.data[newcols]

    @property
    def SISSO_features_dimensions_ranges(self):
        cols = list(self.data.columns)
        if self.model_type == 'regression':
            ii = 2
        elif self.model_type == 'classification':
            ii = 1
        else:
            raise ValueError('Wrong model_type')
        featcols = cols[ii:]
        featdimensions = [self.features_dimensions[featcol]
                          if featcol in self.features_dimensions else None for featcol in featcols]
        uniquedimensions = list(set(featdimensions))
        ranges = {}
        for dimension in uniquedimensions:
            idx = featdimensions.index(dimension)
            count = featdimensions.count(dimension)
            ranges[dimension] = (idx+1, idx+count)
        # Check that the ranges do not overlap
        for dim1, range1 in ranges.items():
            for dim2, range2 in ranges.items():
                if dim1 == dim2:
                    continue
                if self._check_ranges_overlap(range1, range2):
                    raise ValueError("Dimension ranges overlap :")
        return ranges
        # current_dimension = None
        # for featcol in featcols:
        #     if featcol in self.features_dimensions:
        #         dimension =
        #     dimension = self.features_dimensions[featcol] if featcol in self.features_dimensions else '_NODIM'
        # return NotImplementedError

    @staticmethod
    def _check_ranges_overlap(r1, r2):
        return not ((r1[0] < r2[0] and r1[1] < r2[0]) or (r2[0] < r1[0] and r2[1] < r1[0]))


    @property
    def nsample(self):
        return len(self.data)

    @property
    def nsf(self):
        return len(self.data.columns)-2

    @property
    def input_string(self):
        out = [' '.join(['{:20}'.format(column_name) for column_name in self.data.columns])]
        max_str_size = max(self.data[self.data.columns[0]].apply(len))
        header_row_format_str = '{{:{}}}'.format(max(20, max_str_size))
        for _, row in self.data.iterrows():
            row_list = list(row)
            line = [header_row_format_str.format(row_list[0])]
            # line = ['{:20}'.format(row_list[0])]
            for col in row_list[1:]:
                line.append('{:<20.12f}'.format(col))
            out.append(' '.join(line))
        return '\n'.join(out)

    def to_file(self, filename='train.dat'):
        with open(filename, 'w') as f:
            f.write(self.input_string)

    @classmethod
    def from_file(cls, filepath):
        if filepath.endswith('.dat'):
            return cls.from_dat_file(filepath)
        else:
            raise ValueError('The from_file method is working only with .dat files')

    @classmethod
    def from_dat_file(cls, filepath):
        data = pd.read_csv(filepath, delim_whitespace=True)
        return cls(data=data)


class SISSOIn(MSONable):
    """
    Main class containing the input variables for SISSO.

    This class is basically a container for the SISSO.in input file for SISSO.
    Additional helper functions are available.
    """

    #: dict: Types or descriptions (as a string) of the values for each SISSO keyword.
    KW_TYPES = {'ptype': tuple([int]),
                'ntask': tuple([int]),
                'nsample': tuple([int, 'list_of_ints']),
                'task_weighting': tuple([int]),
                'desc_dim': tuple([int]),
                'nsf': tuple([int]),
                'restart': tuple([bool]),
                'rung': tuple([int]),
                'opset': tuple(['str_operators']),
                'maxcomplexity': tuple([int]),
                'dimclass': tuple(['str_dimensions']),
                'maxfval_lb': tuple([float]),
                'maxfval_ub': tuple([float]),
                'subs_sis': tuple([int, 'list_of_ints']),
                'method': tuple([str]),
                'L1L0_size4L0': tuple([int]),
                'fit_intercept': tuple([bool]),
                'metric': tuple([str]),
                'nm_output': tuple([int]),
                'isconvex': tuple(['str_isconvex']),
                'width': tuple([float]),
                'nvf': tuple([int]),
                'vfsize': tuple([int]),
                'vf2sf': tuple([str]),
                'npf_must': tuple([int]),
                'L1_max_iter': tuple([int]),
                'L1_tole': tuple([float]),
                'L1_dens': tuple([int]),
                'L1_nlambda': tuple([int]),
                'L1_minrmse': tuple([float]),
                'L1_warm_start': tuple([bool]),
                'L1_weighted': tuple([bool])
                }

    #: dict: Available unary and binary operators for feature construction. TODO: add string description
    AVAILABLE_OPERATIONS = {'unary': {'exp': '',
                                      'exp-': '',
                                      '^-1': '',
                                      'scd': '',
                                      '^2': '',
                                      '^3': '',
                                      '^6': '',
                                      'sqrt': '',
                                      'cbrt': '',
                                      'log': '',
                                      'sin': '',
                                      'cos': ''
                                      },
                            'binary': {'+': '',
                                       '-': '',
                                       '|-|': '',
                                       '*': '',
                                       '/': ''
                                       }}

    def __init__(self, target_properties_keywords,
                 feature_construction_sure_independence_screening_keywords,
                 descriptor_identification_keywords):
        self.target_properties_keywords = target_properties_keywords
        self.feature_construction_sure_independence_screening_keywords = feature_construction_sure_independence_screening_keywords
        self.descriptor_identification_keywords = descriptor_identification_keywords
        self._check_keywords()

    def _check_keywords(self):
        #TODO: implement a check on the keywords
        pass

    def _format_kw_value(self, kw, val, float_format='.12f'):
        allowed_types = self.KW_TYPES[kw]
        # Determine the type of the value for this keyword
        val_type = None
        for allowed_type in allowed_types:
            if allowed_type is int:
                if type(val) is int:
                    val_type = int
                    break
            elif allowed_type is float:
                if type(val) is float:
                    val_type = float
                    break
            elif allowed_type is bool:
                if type(val) is bool:
                    val_type = bool
                    break
            elif allowed_type is str:
                if type(val) is str:
                    val_type = str
                    break
            elif allowed_type == 'list_of_ints':
                if (type(val) is list or type(val) is tuple) and all([type(item) is int for item in val]):
                    val_type = 'list_of_ints'
                    break
            #TODO: add checks on the str_operators, str_dimensions and str_isconvex
            elif allowed_type == 'str_operators':
                val_type = 'str_operators'
            elif allowed_type == 'str_dimensions':
                val_type = 'str_dimensions'
            elif allowed_type == 'str_isconvex':
                val_type = 'str_isconvex'
        if val_type is None:
            raise ValueError('Type of value "{}" for keyword "{}" not found/valid.'.format(str(val), kw))

        if val_type is int:
            return '{}={:d}'.format(kw, val)
        elif val_type is float:
            float_ref_str = '{}={{:{}}}'.format(kw, float_format)
            return float_ref_str.format(val)
        elif val_type is bool:
            return '{}=.{}.'.format(kw, str(val).lower())
        elif val_type is str:
            return '{}=\'{}\''.format(kw, val)
        elif val_type == 'list_of_ints':
            return '{}=({})'.format(kw, ','.join(['{:d}'.format(v) for v in val]))
        elif val_type == 'str_operators':
            return '{}=\'{}\''.format(kw, val)
        elif val_type in ['str_dimensions', 'str_isconvex']:
            return '{}={}'.format(kw, val)
        else:
            raise ValueError('Wrong type for SISSO value.\nSISSO keyword : {}\n'
                             'Value : {} (type : {})'.format(kw, str(val), val_type))

    @property
    def input_string(self, matgenix_acknowledgement=True):
        if (self.target_properties_keywords['nsample'] is None or
                self.feature_construction_sure_independence_screening_keywords['nsf'] is None):
            raise ValueError('Both keywords "nsample" and "nsf" should be set to get SISSO.in\'s input_string')
        out = []
        if matgenix_acknowledgement:
            year = datetime.datetime.now().year
            out.append('!--------------------------------------------------------!\n'
                       '! SISSO.in generated by Matgenix\'s pysisso package.      !\n'
                       '! Copyright (c) {:d}, Matgenix SRL. All Rights Reserved. !\n'
                       '!--------------------------------------------------------!\n'.format(year))
        if self.is_regression:
            out.append('!------------------!\n'
                       '! REGRESSION MODEL !\n'
                       '!------------------!\n')
        elif self.is_classification:
            out.append('!----------------------!\n'
                       '! CLASSIFICATION MODEL !\n'
                       '!----------------------!\n')

        # Keywords related to target properties
        out.append('!------------------------------------!\n'
                   '! Keywords for the target properties !\n'
                   '!------------------------------------!')
        for sisso_kw, sisso_val in self.target_properties_keywords.items():
            if sisso_val is None:
                continue
            out.append(self._format_kw_value(kw=sisso_kw, val=sisso_val))
        out.append('')

        # Keywords related to feature construction (FC) and sure independence screening (SIS)
        out.append('!------------------------------------------------------------------------------!\n'
                   '! Keywords for feature construction (FC) and sure independence screening (SIS) !\n'
                   '!------------------------------------------------------------------------------!')
        for sisso_kw, sisso_val in self.feature_construction_sure_independence_screening_keywords.items():
            if sisso_val is None:
                continue
            out.append(self._format_kw_value(kw=sisso_kw, val=sisso_val))
        out.append('')

        # Keywords descriptor identification via a sparsifying operator
        out.append('!------------------------------------------------------------------!\n'
                   '! Keyword for descriptor identification via a sparsifying operator !\n'
                   '!------------------------------------------------------------------!')
        for sisso_kw, sisso_val in self.descriptor_identification_keywords.items():
            if sisso_val is None:
                continue
            out.append(self._format_kw_value(kw=sisso_kw, val=sisso_val))
        return '\n'.join(out)

    @property
    def is_regression(self) -> bool:
        """Whether this SISSOIn object corresponds to a regression model.

        Returns:
            bool: True if this SISSOIn object is a regression model, False otherwise.
        """
        return self.target_properties_keywords['ptype'] == 1

    @property
    def is_classification(self):
        """Whether this SISSOIn object corresponds to a classification model.

        Returns:
            bool: True if this SISSOIn object is a classification model, False otherwise.
        """
        return self.target_properties_keywords['ptype'] == 2

    @classmethod
    def from_sisso_keywords(cls, ptype, nsample=None, nsf=None, ntask=1, task_weighting=1, desc_dim=2, restart=False,
                            rung=2, opset='(+)(-)', maxcomplexity=10, dimclass=None,
                            maxfval_lb=1e-3, maxfval_ub=1e5, subs_sis=20,
                            method='L0', L1L0_size4L0=1, fit_intercept=True, metric='RMSE', nm_output=100,
                            isconvex=None, width=None, nvf=None, vfsize=None, vf2sf=None, npf_must=None,
                            L1_max_iter=None, L1_tole=None, L1_dens=None, L1_nlambda=None, L1_minrmse=None,
                            L1_warm_start=None, L1_weighted=None):
        target_properties_keywords = dict()
        target_properties_keywords['ptype'] = ptype
        target_properties_keywords['ntask'] = ntask
        target_properties_keywords['nsample'] = nsample
        target_properties_keywords['task_weighting'] = task_weighting
        target_properties_keywords['desc_dim'] = desc_dim
        target_properties_keywords['restart'] = restart
        feature_construction_sure_independence_screening_keywords = dict()
        feature_construction_sure_independence_screening_keywords['nsf'] = nsf
        feature_construction_sure_independence_screening_keywords['rung'] = rung
        feature_construction_sure_independence_screening_keywords['opset'] = opset
        feature_construction_sure_independence_screening_keywords['maxcomplexity'] = maxcomplexity
        feature_construction_sure_independence_screening_keywords['dimclass'] = dimclass
        feature_construction_sure_independence_screening_keywords['maxfval_lb'] = maxfval_lb
        feature_construction_sure_independence_screening_keywords['maxfval_ub'] = maxfval_ub
        feature_construction_sure_independence_screening_keywords['subs_sis'] = subs_sis
        feature_construction_sure_independence_screening_keywords['nvf'] = nvf
        feature_construction_sure_independence_screening_keywords['vfsize'] = vfsize
        feature_construction_sure_independence_screening_keywords['vf2sf'] = vf2sf
        feature_construction_sure_independence_screening_keywords['npf_must'] = npf_must
        descriptor_identification_keywords = dict()
        descriptor_identification_keywords['method'] = method
        descriptor_identification_keywords['L1L0_size4L0'] = L1L0_size4L0
        descriptor_identification_keywords['fit_intercept'] = fit_intercept
        descriptor_identification_keywords['metric'] = metric
        descriptor_identification_keywords['nm_output'] = nm_output
        descriptor_identification_keywords['isconvex'] = isconvex
        descriptor_identification_keywords['width'] = width
        descriptor_identification_keywords['L1_max_iter'] = L1_max_iter
        descriptor_identification_keywords['L1_tole'] = L1_tole
        descriptor_identification_keywords['L1_dens'] = L1_dens
        descriptor_identification_keywords['L1_nlambda'] = L1_nlambda
        descriptor_identification_keywords['L1_minrmse'] = L1_minrmse
        descriptor_identification_keywords['L1_warm_start'] = L1_warm_start
        descriptor_identification_keywords['L1_weighted'] = L1_weighted
        return cls(target_properties_keywords=target_properties_keywords,
                   feature_construction_sure_independence_screening_keywords=feature_construction_sure_independence_screening_keywords,
                   descriptor_identification_keywords=descriptor_identification_keywords)

    @classmethod
    def from_file(cls, filepath):
        raise NotImplementedError

    def to_file(self, filename='SISSO.in'):
        with open(filename, 'w') as f:
            f.write(self.input_string)

    def set_keywords_for_SISSO_dat(self, sisso_dat):
        dimclass = None
        if sisso_dat.features_dimensions is not None:
            feature_dimensions_ranges = sisso_dat.SISSO_features_dimensions_ranges
            if (len(feature_dimensions_ranges) == 0 or
                    (len(feature_dimensions_ranges) == 1 and list(feature_dimensions_ranges.keys())[0] is None)):
                dimclass = None
            else:
                dimclasslist = []
                for dim, dimrange in feature_dimensions_ranges.items():
                    if dim is None:
                        continue
                    dimclasslist.append('({:d}:{:d})'.format(dimrange[0], dimrange[1]))
                dimclass = ''.join(dimclasslist)

        self.target_properties_keywords['nsample'] = sisso_dat.nsample
        self.feature_construction_sure_independence_screening_keywords['nsf'] = sisso_dat.nsf
        self.feature_construction_sure_independence_screening_keywords['dimclass'] = dimclass

    @classmethod
    def from_SISSO_dat(cls, sisso_dat: SISSODat, model_type: str = 'regression', **kwargs: object):

        if model_type == 'regression':
            ptype = 1
        elif model_type == 'classification':
            raise NotImplementedError
        else:
            raise ValueError('Wrong model_type ("{}"). Should be "regression" or "classification".'.format(model_type))
        sissoin = cls.from_sisso_keywords(ptype=ptype, **kwargs)
        sissoin.set_keywords_for_SISSO_dat(sisso_dat=sisso_dat)
        return sissoin
        # feature_dimensions_ranges = sisso_dat.SISSO_features_dimensions_ranges
        # if (len(feature_dimensions_ranges) == 0 or
        #         (len(feature_dimensions_ranges) == 1 and list(feature_dimensions_ranges.keys())[0] is None)):
        #     dimclass = None
        # else:
        #     dimclasslist = []
        #     for dim, dimrange in feature_dimensions_ranges.items():
        #         if dim is None:
        #             continue
        #         dimclasslist.append('({:d}:{:d})'.format(dimrange[0], dimrange[1]))
        #     dimclass = ''.join(dimclasslist)
        # return cls.from_sisso_keywords(ptype=ptype, nsample=sisso_dat.nsample,
        #                                nsf=sisso_dat.nsf, dimclass=dimclass, **kwargs)

class SISSOPredictPara(MSONable):
    """
    Main class containing the input variables for SISSO_predict.

    This class is basically a container for the SISSO_predict_para input file for SISSO_predict.
    """

    def __init__(self, nsample, nsf, maxdimension, model_type='regression'):
        self.nsample = nsample
        self.nsf = nsf
        self.maxdimension = maxdimension
        self.model_type = model_type

    @property
    def input_string(self):
        if self.model_type == 'regression':
            ptype = 1
        elif self.model_type == 'classification':
            ptype = 2
        else:
            raise ValueError('Input variable "model_type" is "{}" '
                             'while it should be either "regression" or "classification"'.format(self.model_type))
        year = datetime.datetime.now().year
        out = ['{:12d}  ! Number of test-materials in the file predict.dat'.format(self.nsample),
               '{:12d}  ! Number of features in the file predict.dat'.format(self.nsf),
               '{:12d}  ! Highest dimension of the models to be read from SISSO.out'.format(self.maxdimension),
               '{:12d}  ! Property type 1:continuous or 2:categorical'.format(ptype),
               '!-------------------------------------------------------------!\n'
               '! SISSO_predict_para generated by Matgenix\'s pysisso package. !\n'
               '! Copyright (c) {:d}, Matgenix SRL. All Rights Reserved.      !\n'
               '!-------------------------------------------------------------!\n'.format(year)]
        return '\n'.join(out)

    def to_file(self, filename='SISSO_predict_para'):
        with open(filename, 'w') as f:
            f.write(self.input_string)
