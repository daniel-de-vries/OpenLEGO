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
import os

from lxml.etree import _Element, _ElementTree
from openmdao.api import Problem, ScipyOptimizeDriver, pyOptSparseDriver, DOEDriver, UniformGenerator, \
    FullFactorialGenerator, BoxBehnkenGenerator, LatinHypercubeGenerator, ListGenerator, view_model, SqliteRecorder, \
    CaseReader
from openmdao.core.driver import Driver
from typing import Optional, Any, Union, Dict

from openlego.core.model import LEGOModel
from openlego.utils.cmdows_utils import get_element_by_uid, get_opt_setting_safe, get_doe_setting_safe
from openlego.utils.general_utils import CachedProperty, print_optional, add_or_append_dict_entry


class LEGOProblem(Problem):
    """Specialized OpenMDAO Problem class representing the problem specified by a CMDOWS file.

    An important note about this class in the context of OpenMDAO is that the aggregation pattern of the root Problem
    class has is changed into a stronger composition pattern. This is because this class directly
    controls the creation and assembly of this class by making use of Python's @property decorator. It is not possible,
    nor should it be attempted, to manually inject a different instance of Problem in place of this, because the
    correspondence between the CMDOWS file and the Problem can then no longer be guaranteed.

    Attributes
    ----------
        cmdows_path
        kb_path
        case_reader_path
        model_view_path
        drivers
        driver_type
        model
        driver

        data_folder : str, optional
            Path to the folder in which to store all data generated during the `Problem`'s execution.

        base_xml_file : str, optional
            Path to an XML file which should be kept up-to-date with the latest data describing the problem.

        output_case_string : str, optional
            A string indicating the naming for output files such as N2 views and recorders.
    """

    def __init__(self, cmdows_path=None, kb_path='', data_folder=None, base_xml_file=None, output_case_str=None,
                 **kwargs):
        # type: (Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]) -> None
        """Initialize a CMDOWS Problem from a given CMDOWS file and (optionally) knowledge base.

        It is also possible to specify where (temporary) data should be stored, and if a base XML
        file should be kept up-to-date.

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

        output_case_str : str, optional
            A string indicating the naming for output files such as N2 views and recorders.
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

    def __getattribute__(self, name):
        # type: (str) -> Any
        """Check the integrity before returning any of the cached variables.

        Parameters
        ----------
        name : str
            Name of the attribute to read.

        Returns
        -------
            any
                The value of the requested attribute.
        """
        if name != '__class__' and name != '__dict__':
            if name in [_name for _name, value in self.__class__.__dict__.items() if isinstance(value, CachedProperty)]:
                self.__integrity_check()
        return super(LEGOProblem, self).__getattribute__(name)

    def __setattr__(self, name, value):
        # type: (str, Any) -> None
        """Bypass setting model and driver attribute.

        Parameters
        ----------
            name : str
                Name of the attribute.

            value : any
                Value to set the attribute to.
        """
        if name not in ['model', 'driver']:
            super(LEGOProblem, self).__setattr__(name, value)

    def __integrity_check(self):
        # type: () -> None
        """Ensure a CMDOWS file has been supplied.

        Raises
        ------
            ValueError
                If no CMDOWS file has been supplied
        """
        if self._cmdows_path is None:
            raise ValueError('No CMDOWS file specified!')

    def invalidate(self):
        # type: () -> None
        """Invalidate the instance.

        All computed (cached) properties will be recomputed upon being read once the instance has been invalidated."""
        self.cleanup()  # Cleanup the problem
        self.model.invalidate()  # Invalidate the model
        # Invalidate the problem
        for value in self.__class__.__dict__.values():
            if isinstance(value, CachedProperty):
                value.invalidate()

    @property
    def cmdows_path(self):
        # type: () -> str
        """:obj:`str`: Path to the CMDOWS file this class corresponds to.

        When this property is set the instance is automatically invalidated.
        """
        return self._cmdows_path

    @cmdows_path.setter
    def cmdows_path(self, cmdows_path):
        # type: (str) -> None
        self._cmdows_path = cmdows_path
        self.invalidate()

    @property
    def kb_path(self):
        # type: () -> str
        """:obj:`str`: Path to the knowledge base.

        When this property is set the instance is automatically invalidated.
        """
        return self._kb_path

    @kb_path.setter
    def kb_path(self, kb_path):
        # type: (str) -> None
        self._kb_path = kb_path
        self.invalidate()


    @CachedProperty
    def case_reader_path(self):
        # type: () -> str
        """:obj:`str`: Path to the case reader that has been added to the driver."""
        filename = 'case_reader_' + self.output_case_string + '.sql'
        if self.data_folder:
            return os.path.join(self.data_folder, filename)
        else:
            return filename

    @CachedProperty
    def model_view_path(self):
        # type: () -> str
        """:obj:`str`: Path to the model view html view that is created with store_model_view()."""
        filename = 'n2_' + self.output_case_string + '.html'
        if self.data_folder:
            return os.path.join(self.data_folder, filename)
        else:
            return filename

    @CachedProperty
    def drivers(self):
        # type: () -> dict
        """:obj:`dict`: Dictionary containing the optimizer and DOE element UIDs from the CMDOWS file."""
        optimizer_elems = self.model.elem_arch_elems.findall('executableBlocks/optimizers/optimizer')
        doe_elems = self.model.elem_arch_elems.findall('executableBlocks/does/doe')
        optimizers = [elem.get('uID') for elem in optimizer_elems]
        does = [elem.get('uID') for elem in doe_elems]
        return {'optimizers': optimizers, 'does': does}

    @CachedProperty
    def driver_type(self):
        # type: () -> Union[str, None]
        """:obj:`str`: Type of driver as string."""
        if len(self.drivers['optimizers']) + len(self.drivers['does']) > 1:
            raise AssertionError("Only one driver is allowed at the moment. {} drivers specified ({}) at the moment."
                                 .format(len(self.drivers), self.drivers))
        if self.drivers['optimizers']:
            return 'optimizer'
        elif self.drivers['does']:
            return 'doe'
        else:
            return None

    @CachedProperty
    def model(self):
        # type: () -> LEGOModel
        """:obj:`LEGOModel`: The LEGOModel that is automatically built from the CMDOWS file and knowledge base."""
        return LEGOModel(self._cmdows_path,   # CMDOWS file
                         self._kb_path,       # Knowledge base path
                         self.data_folder,    # Output directory
                         self.base_xml_file)  # Output file

    @CachedProperty
    def driver(self):
        # type: () -> Driver
        """Method to return a preconfigured driver.

        Returns
        -------
            Driver
                A preconfigured driver element to be used for the Problem instance.

        Raises
        ------
            ValueError
                Value error are raised if unsupported settings are encountered.
        """
        if self.driver_type:
            if self.driver_type == 'optimizer':
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
                return driver
            elif self.driver_type == 'doe':
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
                doe_data = []
                if isinstance(doe_elem.find('settings/table'), _Element):
                    doe_table = doe_elem.find('settings/table')
                    doe_table_rows = [row for row in doe_table.iterchildren()]
                    n_samples = len([exp for exp in doe_table_rows[0].iterchildren()])
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
                return driver
            else:
                raise ValueError('Driver was found, but no optimizer or doe was found somehow.')
        else:
            return Driver()

    def store_model_view(self, open_in_browser=False):
        # type: (bool) -> None
        """Implementation of the view_model() function for storage and (optionally) viewing in the browser.

        Parameters
        ----------
            open_in_browser : bool
                Setting whether to attempt to automatically open the model view in the browser.
        """
        if self._setup_status == 0:
            self.setup()
        view_model(self, outfile=self.model_view_path, show_browser=open_in_browser)

    def initialize(self):
        # type: () -> None
        """Method to initialize the problem by adding a recorder and doing the setup."""
        self.driver.add_recorder(SqliteRecorder(self.case_reader_path))
        if self._setup_status == 0:
            self.setup()

    def initialize_from_xml(self, xml):
        # type: (Union[str, _ElementTree]) -> None
        """Initialize the problem with initial values from an XML file.

        Parameters
        ----------
            xml : str or :obj:`etree._ElementTree`
                Path to an XML file or an instance of `etree._ElementTree` representing it.
                """
        self.initialize()
        self.run_model()
        self.model.initialize_from_xml(xml)

    def collect_results(self, cases_to_collect='default', print_in_log=True):
        # type: (Union[str, list]) -> Dict[dict]
        """Print the results that were stored in the case reader.

        Parameters
        ----------
            cases_to_collect : str or list
                Setting on which cases should be print (e.g. 'last', 'all', 'default', [2, 3, 5]

            print_in_log : bool
                Setting on whether the results should also be printed in the log

        Returns
        -------
            results : dict of dicts
                Dictionary containing the results that were collected
        """
        if not os.path.isfile(self.case_reader_path):
            raise AssertionError('Could not find the case reader file {}.'.format(self.case_reader_path))
        if not isinstance(cases_to_collect, (str, list)):
            raise AssertionError('cases_to_print must be of type str or list.')
        if isinstance(cases_to_collect, str):
            if cases_to_collect not in ['default', 'all', 'last']:
                raise AssertionError('Invalid cases_to_print string value provided.')
        if cases_to_collect== 'default':
            if self.driver_type == 'doe':
                cases_to_collect = 'all'
            elif self.driver_type == 'optimizer':
                cases_to_collect = 'last'
            else:
                cases_to_collect = 'last'
        results = dict()

        # Get all cases from the case reader and determine amount of cases
        cases = CaseReader(self.case_reader_path).driver_cases
        num_cases = cases.num_cases

        # Change cases_to_print to a list of integers with case numbers
        if isinstance(cases_to_collect, str):
            if cases_to_collect == 'all':
                cases_to_collect = range(num_cases)
            elif cases_to_collect == 'last':
                cases_to_collect = [num_cases - 1]

        # Print results
        print_optional('\nPrinting results from case reader: {}.'.format(self.case_reader_path), print_in_log)
        if self.driver.fail:
            if self.driver_type == 'optimizer':
                print_optional('Optimum not found in driver execution!', print_in_log)
            else:
                print_optional('Driver failed for some reason!', print_in_log)
        else:
            if self.driver_type == 'optimizer':
                print_optional('Optimum found!', print_in_log)
            else:
                print_optional('Driver finished!', print_in_log)
        print_optional('\nPrinting case numbers: {}'.format(cases_to_collect), print_in_log)
        for num_case in cases_to_collect:
            case = cases.get_case(num_case)
            print_optional('\n\n  Case {}/{} ({})'.format(num_case, num_cases-1, case.iteration_coordinate),
                           print_in_log)
            recorded_objectives = case.get_objectives()
            recorded__desvars = case.get_desvars()
            recorded_constraints = case.get_constraints()
            var_objectives = sorted(list(recorded_objectives.keys))
            var_desvars = sorted(list(recorded__desvars.keys))
            var_constraints = sorted(list(recorded_constraints.keys))
            var_does = sorted([elem.text for elem in self.model.elem_arch_elems
                              .findall('parameters/doeOutputSampleLists/doeOutputSampleList/relatedParameterUID')])
            var_convs = sorted([elem.text for elem in self.model.elem_problem_def
                              .findall('problemRoles/parameters/stateVariables/stateVariable/parameterUID')])

            # Print objective
            if var_objectives:
                print_optional('    Objectives', print_in_log)
                for var_objective in var_objectives:
                    value = recorded_objectives[var_objective]
                    print_optional('    {}: {}'.format(var_objective, value), print_in_log)
                    results = add_or_append_dict_entry(results, 'objectives', var_objective, value)

            # Print design variables
            # TODO: Add bounds (currently a bug in OpenMDAO recorders)
            if var_desvars:
                print_optional('\n    Design variables', print_in_log)
                for var_desvar in var_desvars:
                    value = recorded__desvars[var_desvar]
                    print_optional('    {}: {}'.format(var_desvar, value), print_in_log)
                    results = add_or_append_dict_entry(results, 'desvars', var_desvar, value)

            # Print constraint values
            # TODO: Add bounds (currently a bug in OpenMDAO recorders)
            if var_constraints:
                print_optional('\n    Constraints', print_in_log)
                for var_constraint in var_constraints:
                    value = recorded_constraints[var_constraint]
                    print_optional('    {}: {}'.format(var_constraint, value), print_in_log)
                    results = add_or_append_dict_entry(results, 'desvars', var_constraint, value)

            # Print DOE quantities of interest
            if var_does:
                print_optional('\n    Quantifies of interest', print_in_log)
                for var_qoi in var_does:
                    value = case.outputs[var_qoi]
                    print_optional('    {}: {}'.format(var_qoi, value), print_in_log)
                    results = add_or_append_dict_entry(results, 'qois', var_qoi, value)

            # Print other quantities of interest
            title_not_printed = True
            if var_convs:
                for var_qoi in var_convs:
                    if var_qoi not in var_objectives + var_constraints + var_does:
                        if title_not_printed:
                            print_optional('\n    Quantifies of interest', print_in_log)
                            title_not_printed = False
                        value = case.outputs[var_qoi]
                        print_optional('    {}: {}'.format(var_qoi, value), print_in_log)
                        results = add_or_append_dict_entry(results, 'qois', var_qoi, value)
        return results
