#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2018 D. de Vries

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains the definition of general utility functions.
"""
from __future__ import absolute_import, division, print_function

import os
import re
import warnings
from os import path

import numpy as np
from lxml import etree
from typing import Callable, Any, Union, List, SupportsInt, SupportsFloat


def try_hard(fun, *args, **kwargs):
    # type: (Callable, *Any, **Any) -> Any
    """Try repeatedly to call a function until it returns successfully.

    Utility function that repeatedly tries to call a given function with the given arguments until that function
    successfully returns. It is possible to limit the maximum number of attempts by setting the try_hard_limit argument.


    Parameters
    ----------
        fun : function
            The function to try to call.

        *args
            Any ordered arguments to pass to the function.

        **kwargs
            Any named arguments to pass to the function.


    Returns
    -------
        any
            The return value of the function to be called.

    Notes
    -----
        This function is part of a workaround to deal with APIs crashing in a non-predictable manner, seemingly random
        way. It was found that, when simply trying to call such functions again, the problem seemed to not exist
        anymore.
    """
    warnings.simplefilter('always', UserWarning)
    try_hard_limit = -1
    if kwargs is not None:
        if 'try_hard_limit' in kwargs.keys():
            try_hard_limit = kwargs['try_hard_limit']
            del kwargs['try_hard_limit']

    msg = None
    return_value = None
    successful = False
    attempts = 0
    while not successful:
        try:
            return_value = fun(*args, **kwargs)
            successful = True
        except Exception as e:
            attempts += 1

            if msg is None:
                msg = 'Had to try again to evaluate: %s(' % fun.__name__ + ', '.join(['%s' % arg for arg in args])
                if kwargs is not None:
                    msg += ', '.join(['%s=%s' % (key, value) for key, value in kwargs.items()])
                msg += '). The following exception was raised: "%s"' % e.message

            if 0 < try_hard_limit <= attempts:
                raise
            else:
                warnings.warn(msg)

    return return_value


def parse_string(s):
    # type: (str) -> Union[str, np.ndarray, float]
    """Convert a string to a numpy array of floats or float if possible.

    The string is returned unchanged if it cannot be converted to a numpy array of floats or float.

    Parameters
    ----------
        s : str
            String to be converted.

    Returns
    -------
        str or np.ndarray or float
            Parsed string or the string itself.
    """
    v = re.sub(r'[\[\]]', '', s)

    if ',' in v:
        v = v.split(',')
    elif ';' in v:
        v = v.split(';')

    try:
        v = np.atleast_1d(np.array(v, dtype=float))
        if v.size == 1:
            v = v[0]
        return v
    except ValueError:
        return s


def is_float(value):
    """
    Returns whether or not the value represents a continuous value.
    """
    if isinstance(value, float):
        return True

    if isinstance(value, np.ndarray):
        return value.dtype == np.float64

    return False


def print_optional(string, print_in_log):
    # type: (str, bool) -> None
    """Print a statement based on a string and a boolean wheter to really print or not.

    Parameters
    ----------
        string : str
            string to be printed

        print_in_log : bool
            boolean on whether the string should really be printed
    """
    if print_in_log:
        print(string)


def add_or_append_dict_entry(main_dict, main_key, sub_key, value):
    # type: (dict, str, str, Any) -> dict
    """Add a value to a sub-key element of a dictionary, if sub-key entry does not exist yet, then it is first created.

    Parameters
    ----------
        main_dict : dict
            main dictionary containing all results

        main_key : str
            main key of the dictionary where the value should be stored

        sub_key : str
            sub-key withing the main_key dictionary under which the value should be added

        value : Any
            the actual value to be added
    """
    if main_key not in main_dict:
        main_dict[main_key] = dict()
    if sub_key not in main_dict[main_key]:
        main_dict[main_key][sub_key] = [value]
    else:
        main_dict[main_key][sub_key].append(value)
    return main_dict


def parse_cmdows_value(elem):
    # type: (etree._Element) -> Union[str, np.ndarray, float]
    """Convert an XML element from a CMDOWS file to a value.

    Parameters
    ----------
        elem : :obj:`_Element`
            `etree._Element` to convert.

    Returns
    -------
        str or np.ndarray or float
            Converted element.
    """
    if len(list(elem)) > 1:
        return np.array([parse_string(child.text) for child in elem])
    else:
        return parse_string(elem.text)


def unscale_value(v, ref0, ref):
    # TODO: add docstring
    if isinstance(v, list):
        v = np.array(v)
    return v*(ref-ref0)+ref0


def scale_value(v, adder, scaler):
    # TODO: add docstring
    if adder is None:
        adder = 0.
    if scaler is None:
        scaler = 1.
    if isinstance(v, list):
        v = np.array(v)
    return (v + adder) * scaler


re_sys_name_char = re.compile(r'[^_a-zA-Z0-9]')
re_sys_name_starts = re.compile(r'^[a-zA-Z]')


def str_to_valid_sys_name(string):
    # type: (str) -> str
    """Ensures a given string is a valid OpenMDAO system name."""
    sys_name = re_sys_name_char.sub('', string)
    while not re_sys_name_starts.match(sys_name):
        sys_name = sys_name[1:]
    return sys_name


def change_object_type(obj, new_type):
    # type: (Union[str, SupportsInt, SupportsFloat], str) -> Union[str, int, float]
    """Attempts to change an object (usually a string) to a different object type."""
    if new_type == 'str':
        return str(obj)
    elif new_type == 'int':
        return int(obj)
    elif new_type == 'float':
        return float(obj)
    else:
        raise IOError('expected_type "{}" is not supported in this function.'.format(new_type))


def clean_dir_filtered(dr, filters):
    # type: (path, List[str]) -> None
    """Removes files in a directory that contain the strings provided in the filter."""
    for f in os.listdir(dr):
        for fltr in filters:
            if fltr in f:
                os.remove(f)
                continue


def warn_about_failed_experiments(failed_experiments):
    # TODO: Add docstring
    if failed_experiments:
        for sm_uid, failure_data in failed_experiments.items():
            if failure_data[1] == 1.:
                raise AssertionError('All experiments failed for surrogate model {}'.format(sm_uid))
            elif failure_data[1] > 0.5:
                warnings.warn('ATTENTION! More than 50% of the experiments (actually {:.1f}%) failed for surrogate'
                              ' model {}'.format(failure_data[1]*100., sm_uid))
            elif failure_data[1] > 0.2:
                warnings.warn('More than 20% of the experiments (actually {:.1f}%) failed for surrogate model {}'
                              .format(failure_data[1] * 100., sm_uid))

            else:
                print('{:.1f}% of the experiments failed for surrogate model {}'.format(failure_data[1] * 100., sm_uid))



class PyOptSparseImportError(ImportError):

    def __init__(self):
        super(PyOptSparseImportError, self).__init__()

    @property
    def msg(self):
        return "Cannot import name pyOptSparseDriver. This probably means that this package has " \
               "not been installed to your Python packages. Note that it needs to be installed to " \
               "your Python manually (no PyPIdistribution available). pyOptSparse can be " \
               "downloaded here: https://github.com/mdolab/pyoptsparse"


def pyoptsparse_installed():
    # type: () -> bool
    """Check for the installation of the PyOptSparse Python package."""
    try:
        from openmdao.api import pyOptSparseDriver
    except ImportError:
        print(PyOptSparseImportError().msg)
        return False
    return True

