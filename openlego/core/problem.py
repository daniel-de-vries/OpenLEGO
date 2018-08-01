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

This file contains the definition of the `LEGOModel` class.
"""
from __future__ import absolute_import, division, print_function

import imp
import re
import warnings

import numpy as np
from lxml import etree
from lxml.etree import _Element, _ElementTree
from openmdao.api import Group, IndepVarComp, LinearBlockGS, NonlinearBlockGS, LinearBlockJac, NonlinearBlockJac, \
    LinearRunOnce, ExecComp, NonlinearRunOnce, Problem, ScipyOptimizeDriver
from typing import Union, Optional, List, Any, Dict, Tuple

from openlego.core.model import LEGOModel
from openlego.utils.general_utils import CachedProperty, parse_cmdows_value, str_to_valid_sys_name, parse_string
from openlego.utils.xml_utils import xpath_to_param, xml_to_dict
from .abstract_discipline import AbstractDiscipline
from .discipline_component import DisciplineComponent


class InvalidCMDOWSFileError(ValueError):

    def __init__(self, reason=None):
        msg = 'Invalid CMDOWS file'
        if reason is not None:
            msg += ': {}'.format(reason)
        super(InvalidCMDOWSFileError, self).__init__(msg)


class LEGOProblem(Problem):
    """Specialized OpenMDAO Problem class representing the problem specified by a CMDOWS file.

    An important note about this class in the context of OpenMDAO is that the aggregation pattern of the root Group
    class the base Problem class has is changed into a stronger composition pattern. This is because this class directly
    controls the creation and assembly of this class by making use of Python's @property decorator. It is not possible,
    nor should it be attempted, to manually inject a different instance of Group  in place of these, because the
    correspondence between the CMDOWS file and the Problem can then no longer be guaranteed.

    Attributes
    ----------
        cmdows_path
        kb_path
        discipline_components
        block_order
        coupled_blocks
        system_order
        system_variables
        system_inputs
        driver
        coordinator

        data_folder : str, optional
            Path to the folder in which to store all data generated during the `Problem`'s execution.

        base_xml_file : str, optional
            Path to an XML file which should be kept up-to-date with the latest data describing the problem.
    """

    def __init__(self, cmdows_path=None, kb_path='', data_folder=None, base_xml_file=None, **kwargs):
        # type: (Optional[str], Optional[str], Optional[str], Optional[str]) -> None
        """Initialize a CMDOWS Problem from a given CMDOWS file and knowledge base.

        It is also possible to specify where (temporary) data should be stored, and if a base XML
        file should be kept up-to-data.

        Parameters
        ----------
        cmdows_path : str, optional
            Path to the CMDOWS file.

        kb_path : str, optional
            Path to the knowledge base.

        data_folder : str, optional
            Path to the data folder in which to store all files and output from the problem.

        base_xml_file : str, optional
            Path to a base XML file to update with the problem data.
        """
        self._cmdows_path = cmdows_path
        self._kb_path = kb_path
        self.data_folder = data_folder
        self.base_xml_file = base_xml_file

        super(LEGOProblem, self).__init__(**kwargs)
        model = self.model = LEGOModel(self._cmdows_path,  # CMDOWS file
                                       self._kb_path,  # Knowledge base path
                                       self.data_folder,  # Output directory
                                        self.base_xml_file)  # Output file
        #self.linear_solver = LinearRunOnce()
        #self.nonlinear_solver = NonlinearRunOnce()
        self.set_solver_print(0)

        driver = self.driver = ScipyOptimizeDriver()  # Use a SciPy for the optimization
        driver.options['optimizer'] = 'SLSQP'  # Use the SQP algorithm
        driver.options['disp'] = True  # Print the result
        driver.opt_settings = {'disp': True, 'iprint': 2}  # Display iterations
        self.setup()
