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

import datetime
import imp
import os
import re
import warnings

import numpy as np
from lxml import etree
from lxml.etree import _Element, _ElementTree
from openmdao.api import Group, IndepVarComp, LinearBlockGS, NonlinearBlockGS, LinearBlockJac, NonlinearBlockJac, \
    LinearRunOnce, ExecComp, NonlinearRunOnce, Problem, ScipyOptimizeDriver, pyOptSparseDriver, DOEDriver, \
    UniformGenerator, FullFactorialGenerator, BoxBehnkenGenerator, LatinHypercubeGenerator, ListGenerator, view_model, \
    SqliteRecorder
from openmdao.core.driver import Driver
from typing import Union, Optional, List, Any, Dict, Tuple

from openlego.core.model import LEGOModel
from openlego.utils.cmdows_utils import get_element_by_uid, get_opt_setting_safe, get_doe_setting_safe
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

    def __init__(self, cmdows_path=None, kb_path='', data_folder=None, base_xml_file=None, output_case_str=None, **kwargs):
        # type: (Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]) -> None
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
        if output_case_str:
            self.output_case_string = output_case_str
        else:
            self.output_case_string = os.path.splitext(os.path.basename(cmdows_path))[0] + '_' + \
                                      datetime.datetime.now().isoformat()

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

    def invalidate(self):
        # type: () -> None
        """Invalidate the instance.

        All computed (cached) properties will be recomputed upon being read once the instance has been invalidated."""
        for value in self.__class__.__dict__.values():
            if isinstance(value, CachedProperty):
                value.invalidate()
        # Also invalidate the model
        self.model.invalidate()

    @CachedProperty
    def case_reader_path(self):
        filename = 'case_reader_' + self.output_case_string + '.sql'
        if self.data_folder:
            return os.path.join(self.data_folder, filename)
        else:
            return filename

    @CachedProperty
    def model_view_path(self):
        filename = 'n2_' + self.output_case_string + '.html'
        if self.data_folder:
            return os.path.join(self.data_folder, filename)
        else:
            return filename

    @property
    def drivers(self):
        optimizer_elems = self.model.elem_arch_elems.findall('executableBlocks/optimizers/optimizer')
        doe_elems = self.model.elem_arch_elems.findall('executableBlocks/does/doe')
        optimizers = [elem.get('uID') for elem in optimizer_elems]
        does = [elem.get('uID') for elem in doe_elems]
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
            assert len(self.drivers['optimizers']) + len(self.drivers['does']) <= 1, \
                "Only one driver is allowed at the moment. {} drivers specified ({}) at the moment."\
                    .format(len(self.drivers), self.drivers)
            if self.model.has_optimizer:
                # Find optimizer element in CMDOWS file
                opt_uid = self.drivers['optimizers'][0]
                opt_elem = get_element_by_uid(self.model.elem_cmdows, opt_uid)
                # Load settings from CMDOWS file
                opt_package = get_opt_setting_safe(opt_elem, 'package', 'SciPy')
                opt_algo = get_opt_setting_safe(opt_elem, 'algorithm', 'SLSQP')
                opt_maxiter = get_opt_setting_safe(opt_elem, 'maximumIterations', 50, expected_type='int')
                opt_convtol = get_opt_setting_safe(opt_elem, 'convergenceTolerance', 1e-6, expected_type='float')

                # Apply settings to the driver
                # driver
                if opt_package == 'SciPy':
                    driver = ScipyOptimizeDriver()
                elif opt_package == 'pyOptSparse':
                    driver = pyOptSparseDriver()
                else:
                    raise ValueError('Unsupported package {} encountered in CMDOWS file for optimizer "{}".'
                                     .format(opt_package, opt_uid))

                # optimization algorithm
                if opt_algo == 'SLSQP':
                    driver.options['optimizer'] = 'SLSQP'
                else:
                    raise ValueError('Unsupported algorithm {} encountered in CMDOWS file for optimizer "{}".'
                                     .format(opt_algo, opt_uid))

                # maximum iterations and tolerance
                if isinstance(driver, ScipyOptimizeDriver):
                    driver.options['maxiter'] = opt_maxiter
                    driver.options['tol'] = opt_convtol
                elif isinstance(driver, pyOptSparseDriver):
                    driver.opt_settings['MAXIT'] = opt_maxiter
                    driver.opt_settings['ACC'] = opt_convtol

                # Set default display and output settings
                if isinstance(driver, ScipyOptimizeDriver):
                    driver.options['disp'] = True  # Print the result
                driver.options['debug_print'] = ['desvars', 'nl_cons', 'ln_cons', 'objs']
                driver.add_recorder(SqliteRecorder(self.case_reader_path))
                return driver
            elif self.model.has_doe:
                # Find DOE element in CMDOWS file
                doe_uid = self.drivers['does'][0]
                doe_elem = get_element_by_uid(self.model.elem_cmdows, doe_uid)
                # Load settings from CMDOWS file
                doe_method = get_doe_setting_safe(doe_elem, 'method', 'Uniform design')
                doe_runs = get_doe_setting_safe(doe_elem, 'runs', 5, expected_type='int', doe_method=doe_method,
                                                required_for_doe_methods=['Latin hypercube design', 'Uniform design',
                                                                          'Monte Carlo design'])
                doe_center_runs = get_doe_setting_safe(doe_elem, 'centerRuns', 2, expected_type='int',
                                                       doe_method=doe_method,
                                                       required_for_doe_methods=['Box-Behnken design'])
                doe_seed = get_doe_setting_safe(doe_elem, 'seed', 0, expected_type='int', doe_method=doe_method,
                                                required_for_doe_methods=['Latin hypercube design', 'Uniform design',
                                                                          'Monte Carlo design'])
                doe_levels = get_doe_setting_safe(doe_elem, 'levels', 2, expected_type='int', doe_method=doe_method,
                                                  required_for_doe_methods=['Full factorial design'])

                # table
                if isinstance(doe_elem.find('settings/table'), _Element):
                    doe_table = doe_elem.find('settings/table')
                    doe_table_rows = [row for row in doe_table.iterchildren()]
                    n_samples = len([exp for exp in doe_table_rows[0].iterchildren()])
                    doe_data = []
                    for idx in range(n_samples):
                        data_sample = []
                        for row_elem in doe_table_rows:
                            value = float(row_elem.find('tableElement[@experimentID="{}"]'.format(idx)).text)
                            data_sample.append([row_elem.attrib['relatedParameterUID'], value])
                        doe_data.append(data_sample)
                else:
                    if doe_method in ['Custom design table']:
                        raise ValueError('Table element with data for custom design table missing in CMDOWS file.')

                # Apply settings to the driver
                # define driver
                driver = DOEDriver()

                # define generator
                if doe_method in ['Uniform design', 'Monte Carlo design']:
                    driver.options['generator'] = UniformGenerator(num_samples=doe_runs, seed=doe_seed)
                elif doe_method == 'Full factorial design':
                    driver.options['generator'] = FullFactorialGenerator(levels=doe_levels)
                elif doe_method == 'Box-Behnken design':
                    driver.options['generator'] = BoxBehnkenGenerator(center=doe_center_runs)
                elif doe_method == 'Latin hypercube design':
                    driver.options['generator'] = LatinHypercubeGenerator(samples=doe_runs, seed=doe_seed)
                elif doe_method == 'Custom design table':
                    driver.options['generator'] = ListGenerator(data=doe_data)
                else:
                    raise ValueError('Could not match the doe_method {} with supported methods from OpenMDAO.'
                                     .format(doe_method))
                # settings from OpenMDAO
                driver.options['debug_print'] = ['desvars', 'nl_cons', 'ln_cons', 'objs']
                # add a standard case recorder
                driver.add_recorder(SqliteRecorder(self.case_reader_path))
                return driver
            else:
                raise ValueError('Driver was found, but no optimizer or doe was found somehow.')
        else:
            return Driver()

    def store_model_view(self, open_in_browser=False):
        """Implementation of the view_model() function for storage and (optionally) viewing in the browser."""
        if self._setup_status == 0:
            self.setup()
        view_model(self, outfile=self.model_view_path, show_browser=open_in_browser)

    def initialize_from_xml(self, xml):
        if self._setup_status == 0:
            self.setup()
        self.run_model()
        self.model.initialize_from_xml(xml)
