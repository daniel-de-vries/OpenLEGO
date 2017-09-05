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

This file contains the definition of general utility functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

from typing import Callable, Any, Optional


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
        except:
            attempts += 1

            if msg is None:
                msg = 'Had to try again to evaluate: %s(' % fun.__name__ + ', '.join(['%s' % arg for arg in args])
                if kwargs is not None:
                    msg += ', '.join(['%s=%s' % (key, value) for key, value in kwargs.items()])
                msg += ')'

            if 0 < try_hard_limit <= attempts:
                raise
            else:
                warnings.warn(msg)

    return return_value


class CachedProperty(property):
    """Subclass of `property` using a cache to avoid recalculating an expensive `property` every time it is read.

    An attribute can be decorated with this class is the same way as with a normal `property`. It adds the possibility
    to invalidate the cache when necessary.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        # type: (Optional[Callable], Optional[Callable], Optional[Callable], Optional[str]) -> None
        """Initialize the `CachedProperty`.

        Parameters
        ----------
            fget, fset, fdel : function, optional
                Getter, setter, and deleter functions.

            doc : str, optional
                Docstring of the property.
        """
        super(CachedProperty, self).__init__(fget, fset, fdel, doc)
        self.__cache = None
        self.__dirty = True

    def __get__(self, instance, owner=None):
        # type: (Any, Optional[type]) -> Any
        """Get the value of the property.

        This method checks if the cache of this property is still valid first. If it is, it simply returns the cached
        value. If it isn't, it calls the `super()` to recompute the cached variable, stores it, and then returns the
        newly calculated value.

        Parameters
        ----------
            instance : any
                The instance through which the attribute is accessed.

            owner : type, optional
                The owner of the attribute.

        Returns
        -------
            any
                The value of the attribute.
        """
        if self.__dirty:
            self.__cache = super(CachedProperty, self).__get__(instance, owner)
            self.__dirty = False
        return self.__cache

    def invalidate(self):
        # type: () -> None
        """Marks the cache of the property as invalid, prompting its recomputation the next time it is accessed."""
        self.__dirty = True
