#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2018 I. van Gent and D. de Vries

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains the definition of the `LEGOProblem` class.
"""
from __future__ import absolute_import, division, print_function

import imp
import re
import warnings

import numpy as np
from lxml import etree
from lxml.etree import _Element, _ElementTree
from openmdao.api import Group, IndepVarComp, LinearBlockGS, NonlinearBlockGS, LinearBlockJac, NonlinearBlockJac, \
    LinearRunOnce, ExecComp, NonlinearRunOnce, Problem, ScipyOptimizeDriver, pyOptSparseDriver, DOEDriver
from openmdao.core.driver import Driver
from typing import Union, Optional, List, Any, Dict, Tuple

from openlego.core.model import LEGOModel
from openlego.utils.cmdows_utils import get_element_by_uid
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
    nor should it be attempted, to manually inject a different instance of Group in place of these, because the
    correspondence between the CMDOWS file and the Problem can then no longer be guaranteed.

    Attributes TODO: Update!
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

    def __setattr__(self, name, value):
        # type: (str, Any) -> None
        """Bypass setting model attribute.

        Parameters
        ----------
            name : str
                Name of the attribute.

            value : any
                Value to set the attribute to.
        """
        if name not in ['model', 'driver']:
            super(LEGOProblem, self).__setattr__(name, value)

    @property
    def drivers(self):
        optimizer_elems = self.model.elem_arch_elems.find('executableBlocks/optimizers/optimizer')
        doe_elems = self.model.elem_arch_elems.find('executableBlocks/does/doe')
        if isinstance(optimizer_elems, _Element):
            optimizers = [optimizer_elems.get('uID')]
        elif isinstance(optimizer_elems, list):
            optimizers = [elem.get('uID') for elem in optimizer_elems]
        else:
            optimizers = []
        if isinstance(doe_elems, _Element):
            does = [doe_elems.get('uID')]
        elif isinstance(doe_elems, list):
            does = [elem.get('uID') for elem in doe_elems]
        else:
            does = []
        return {'optimizers': optimizers, 'does': does}

    @CachedProperty
    def model(self):
        return LEGOModel(self._cmdows_path,   # CMDOWS file
                         self._kb_path,       # Knowledge base path
                         self.data_folder,    # Output directory
                         self.base_xml_file)  # Output file

    @CachedProperty
    def driver(self):
        if self.model.has_driver:
            # TODO: Test this part...
            assert len(self.drivers['optimizers']) + len(self.drivers['does']) <= 1, \
                "Only one driver is allowed at the moment. {} drivers specified ({}) at the moment."\
                    .format(len(self.drivers), self.drivers)
            if self.model.has_optimizer:
                opt_uid = self.drivers['optimizers'][0]
                opt_elem = get_element_by_uid(self.model.elem_cmdows, opt_uid)
                # Load settings from CMDOWS file
                # package
                if isinstance(opt_elem.find('settings/package'), _Element):
                    opt_package = opt_elem.find('settings/package').text
                else:
                    warnings.warn('Package not specified for optimizer {}, setting to default SciPy.')
                    opt_package = 'SciPy'

                # algorithm
                if isinstance(opt_elem.find('settings/algorithm'), _Element):
                    opt_algo = opt_elem.find('settings/algorithm').text
                else:
                    warnings.warn('Algorithm not specified for optimizer {}, setting to default SLSQP.')
                    opt_algo = 'SLSQP'

                # maximum iterations
                if isinstance(opt_elem.find('settings/maximumIterations'), _Element):
                    opt_maxiter = int(opt_elem.find('settings/maximumIterations').text)
                else:
                    warnings.warn('Maximum iterations not specified for optimizer {}, setting to default 50.')
                    opt_maxiter = 50

                # convergence tolerance
                if isinstance(opt_elem.find('settings/convergenceTolerance'), _Element):
                    opt_convtol = float(opt_elem.find('settings/convergenceTolerance').text)
                else:
                    warnings.warn('Convergence tolerance not specified for optimizer {}, setting to default 1e-6.')
                    opt_convtol = 1e-6

                # Apply settings to the driver
                if opt_package == 'SciPy':
                    driver = ScipyOptimizeDriver()
                elif opt_package == 'pyOptSparse':
                    driver = pyOptSparseDriver()
                else:
                    raise ValueError('Unsupported package {} encountered in CMDOWS file.'.format(opt_package))

                if opt_algo == 'SLSQP':
                    driver.options['optimizer'] = 'SLSQP'
                else:
                    raise ValueError('Unsupported algorithm {} encountered in CMDOWS file.'.format(opt_package))

                if isinstance(driver, ScipyOptimizeDriver):
                    driver.options['maxiter'] = opt_maxiter
                    driver.options['tol'] = opt_convtol
                elif isinstance(driver, pyOptSparseDriver):
                    driver.opt_settings['MAXIT'] = opt_maxiter
                    driver.opt_settings['ACC'] = opt_convtol

                # Default display and output settings
                driver.options['disp'] = True  # Print the result
                driver.opt_settings = {'disp': True, 'iprint': 2}  # Display iterations
                driver.options['debug_print'] = ['desvars', 'nl_cons', 'ln_cons', 'objs']
                return driver
            elif self.model.has_doe:
                # TODO: Develop this part...
                # doe sampling
                driver = DOEDriver()
                # settings from OpenMDAO
                return driver
            else:
                pass# raise error / exception
        else:
            return Driver()

    def store_model(self, open_in_browser=False):
        """Implementation of the view_model() function for storage and (optionally) viewing in the browser."""

    #
    # def attach_recorders(self):
    #     """Implementation where recorders are automatically attached based on the driver/CMDOWS file."""
