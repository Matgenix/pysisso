# -*- coding: utf-8 -*-
# Copyright (c) 2020, Matgenix SRL


from monty.json import MSONable
import datetime


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
                'width': tuple([float])
                }

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

    def __init__(self, ptype=1, ntask=1, nsample=5, task_weighting=1, desc_dim=2, restart=False,
                 nsf=3, rung=2, opset='(+)(-)', maxcomplexity=10, dimclass='(1:2)(3:3)',
                 maxfval_lb=1e-3, maxfval_ub=1e5, subs_sis=20,
                 method='L0', L1L0_size4L0=1, fit_intercept=True, metric='RMSE', nm_output=100,
                 isconvex=None, width=None):
        self.target_properties_keywords = dict()
        self.target_properties_keywords['ptype'] = ptype
        self.target_properties_keywords['ntask'] = ntask
        self.target_properties_keywords['nsample'] = nsample
        self.target_properties_keywords['task_weighting'] = task_weighting
        self.target_properties_keywords['desc_dim'] = desc_dim
        self.target_properties_keywords['restart'] = restart
        self.feature_construction_sure_independence_screening_keywords = dict()
        self.feature_construction_sure_independence_screening_keywords['nsf'] = nsf
        self.feature_construction_sure_independence_screening_keywords['rung'] = rung
        self.feature_construction_sure_independence_screening_keywords['opset'] = opset
        self.feature_construction_sure_independence_screening_keywords['maxcomplexity'] = maxcomplexity
        self.feature_construction_sure_independence_screening_keywords['dimclass'] = dimclass
        self.feature_construction_sure_independence_screening_keywords['maxfval_lb'] = maxfval_lb
        self.feature_construction_sure_independence_screening_keywords['maxfval_ub'] = maxfval_ub
        self.feature_construction_sure_independence_screening_keywords['subs_sis'] = subs_sis
        self.descriptor_identification_keywords = dict()
        self.descriptor_identification_keywords['method'] = method
        self.descriptor_identification_keywords['L1L0_size4L0'] = L1L0_size4L0
        self.descriptor_identification_keywords['fit_intercept'] = fit_intercept
        self.descriptor_identification_keywords['metric'] = metric
        self.descriptor_identification_keywords['nm_output'] = nm_output
        self.descriptor_identification_keywords['isconvex'] = isconvex
        self.descriptor_identification_keywords['width'] = width

    def _format_kw_value(self, kw, val, float_format='.12f'):
        # print('FORMATTING KEYWORD AND VALUE')
        # print(kw)
        # print(val)
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
    def is_regression(self):
        return self.target_properties_keywords['ptype'] == 1

    @property
    def is_classification(self):
        return self.target_properties_keywords['ptype'] == 2

    @classmethod
    def from_sisso_keywords(cls, ptype=1, ntask=1, nsample=5, task_weighting=1, desc_dim=2, restart=False,
                            nsf=3, rung=2, opset='(+)(-)...', maxcomplexity=10, dimclass='(1:2)(3:3)',
                            maxfval_lb=1e-3, maxfval_ub=1e5, subs_sis=20,
                            method='L0', L1L0_size4L0=1, fit_intercept=True, metric='RMSE', nm_output=100,
                            isconvex=None, width=None):
        raise NotImplementedError

    @classmethod
    def from_file(cls, filepath):
        raise NotImplementedError


class SISSODat(MSONable):
    """Main class containing the data for SISSO (training data, test data or new data).
    """

    def __init__(self, data):
        pass
