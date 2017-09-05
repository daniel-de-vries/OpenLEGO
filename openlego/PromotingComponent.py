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

This file contains the definition of the `PromotingComponent` class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from openmdao.api import Component
from typing import List


class PromotingComponent(Component):
    """Specialized `Component` adding convenience methods to list all in- and/or outputs as a plain list."""
    __metaclass__ = abc.ABCMeta

    def list_inputs(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: List of all ``param`` names."""
        if isinstance(self.params, dict):
            return list(self.params.keys())
        else:
            return list(self._init_params_dict.keys())

    def list_outputs(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: List of all ``unknown`` names."""
        if isinstance(self.unknowns, dict):
            return list(self.unknowns.keys())
        else:
            return list(self._init_unknowns_dict.keys())

    def list_variables(self):
        # type: () -> List[str]
        """:obj:`str`: List of all ``param`` and ``unknown`` names."""
        variables = self.list_inputs()
        variables.extend(self.list_outputs())
        return variables

    @abc.abstractmethod
    def solve_nonlinear(self, params, unknowns, resids):
        super(PromotingComponent, self).solve_nonlinear(params, unknowns, resids)
