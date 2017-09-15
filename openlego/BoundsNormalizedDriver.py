#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2017 D. de Vries

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains the definition of a decorator normalizing design variables of ``OpenMDAO`` `Drivers`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from openmdao.core.driver import Driver
from typing import Type, Optional, Union, Callable


def normalized_to_bounds(driver):
    # type: (Type[Driver]) -> Type[NormalizedDriver]
    """Decorate a `Driver` to adjust its ``adder``/``scaler`` attributes normalizing the ``desvar``s.

    This decorator automatically adjusts the adder and scalar attributes of the design variables belonging to the
    targeted ``OpenMDAO`` `Driver` class such that the design variables are normalized to their bounds.

    Parameters
    ----------
        :obj:`Driver`
            `Driver` to normalize the design variables of.

    Returns
    -------
        :obj:`NormalizedDriver`
            Instance of `NormalizedDriver` which inherits from the given `Driver`.

    Examples
    --------
        @normalized_to_bounds\n
        class MyNormalizedDriver(Driver):
            # My design variables will now automatically be normalized to their bounds.
            pass
    """

    class NormalizedDriver(driver):
        """Wrapper class for the `normalized_to_bounds` decorator.

        This class adds a static function to the `Driver` it inherits from, which will intercept all `add_desvar()`
        calls to the wrapped `Driver` class to change its ``adder``/``scaler`` attributes depending on the given
        upper and lower bounds.
        """

        @staticmethod
        def normalize_to_bounds(func):
            # type: (Callable) -> Callable
            """Wrap the function handle of the `add_desvar()` function.

            Parameters
            ----------
                func : function
                    Function handle of the `add_desvar()` function.

            Returns
            -------
                func : function
                    Function wrapping the `add_desvar()` function, which adds logic to calculate ``adder``/``scaler``.
            """

            def new_function(name,          # type: str
                             lower=None,    # type: Optional[Union[float, np.ndarray]]
                             upper=None,    # type: Optional[Union[float, np.ndarray]]
                             *args, **kwargs):
                # type: (...) -> None
                """Wrap the `add_desvar()` function call.

                Inner wrapper function which will set the ``adder`` and ``scaler`` `kwargs` of the wrapped
                `add_desvar()` method before calling it.

                Parameters
                ----------
                    name : str
                        Name of the design variable to add.

                    lower : float or list of float, optional
                        Lower bound(s) of the design variable.

                    upper : float or list of float, optional
                        Upper bound(s) of the design variable.

                    *args
                        Any extra, ordered arguments to pass to the `add_desvar()` method.

                    **kwargs
                        Any extra, named arguments to pass to the `add_desvar()` method.
                """
                if lower is not None:
                    adder = -lower
                else:
                    adder = 0.

                if upper is not None:
                    scaler = 1./(upper + adder)
                else:
                    scaler = 1.

                if len(args) > 4:
                    args = args[:-1]
                elif len(args) > 3:
                    args = args[:-1]

                if 'adder' in kwargs:
                    del kwargs['adder']
                if 'scaler' in kwargs:
                    del kwargs['scaler']

                func(name, lower, upper, adder=adder, scaler=scaler, *args, **kwargs)

            return new_function

        def __getattribute__(self, item):
            """Intercept any calls to the `add_desvar()` method of the Driver class.

            This ``hook`` checks if `add_desvar()` is called. If so, it returns the wrapped function instead of the
            clean `add_desvar()` call.

            Parameters
            ----------
                item : str
                    Name of the attribute.

            Returns
            -------
                any
                    The attribute that was requested or the wrapped call to `add_desvar()` if it is requested.
            """
            x = super(NormalizedDriver, self).__getattribute__(item)
            if item in ['add_desvar']:
                return self.normalize_to_bounds(x)
            else:
                return x

    return NormalizedDriver
