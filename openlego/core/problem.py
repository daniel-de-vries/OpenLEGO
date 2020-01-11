#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2019 I. van Gent and D. de Vries

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

from cached_property import cached_property
import datetime
from lxml.etree import _Element, _ElementTree
import os
import numpy as np
from typing import Optional, Any, Union, Dict
import warnings


from openmdao.api import Problem, ScipyOptimizeDriver, DOEDriver, UniformGenerator, \
    FullFactorialGenerator, BoxBehnkenGenerator, LatinHypercubeGenerator, ListGenerator, \
    n2, SqliteRecorder, CaseReader
from openmdao.core.driver import Driver

from openlego.core.model import LEGOModel
from openlego.core.cmdows import CMDOWSObject
from openlego.utils.cmdows_utils import get_element_by_uid, get_opt_setting_safe, \
    get_doe_setting_safe
from openlego.utils.general_utils import print_optional, add_or_append_dict_entry, \
    PyOptSparseImportError
from openlego.utils.xml_utils import xml_to_dict, xpath_to_param


class LEGOProblem(CMDOWSObject, Problem):
    """Specialized OpenMDAO Problem class representing the problem specified by a CMDOWS file.

    An important note about this class in the context of OpenMDAO is that the aggregation pattern of
    the root Problem class has is changed into a stronger composition pattern. This is because this
    class directly controls the creation and assembly of this class by making use of Python's
    @property decorator. It is not possible, nor should it be attempted, to manually inject a
    different instance of Problem in place of this, because the correspondence between the CMDOWS
    file and the Problem can then no longer be guaranteed.

    Parameters
    ----------
        cmdows_path : str, optional
            Path to the CMDOWS file.

        kb_path : str, optional
            Path to the knowledge base.

        driver_uid : str, optional
            UID of the main driver under consideration.

        data_folder : str, optional
            Path to the data folder in which to store all files and output from the problem.

        base_xml_file : str, optional
            Path to a base XML file to be updated with the problem data.

        output_case_str : str, optional
            A string used for naming for output files such as N2 (model) views and recorders.

    Returns
    -------
        LEGOProblem
    """

    def __init__(self, cmdows_path=None, kb_path='', driver_uid=None, data_folder=None,
                 base_xml_file=None, output_case_str=None, **kwargs):
        # type: (Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Any) -> None
        """Initialize a CMDOWS Problem from a given CMDOWS file, knowledge base (optional) and
        driver UID (optional).

        It is also possible to specify where (temporary) data should be stored, and if a base XML
        file should be kept up-to-date.
        """
        if output_case_str:
            self.output_case_string = output_case_str
        elif driver_uid:
            self.output_case_string = os.path.splitext(os.path.basename(cmdows_path))[0] + '_' + \
                                      driver_uid + '_' + \
                                      datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-4]
        else:
            self.output_case_string = os.path.splitext(os.path.basename(cmdows_path))[0] + '_' + \
                                      datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-4]
        super(LEGOProblem, self).__init__(cmdows_path, kb_path, driver_uid, data_folder,
                                          base_xml_file, **kwargs)

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

    def invalidate(self):
        # type: () -> None
        """Invalidate the instance.

        All computed (cached) properties will be recomputed upon being read once the instance has
        been invalidated."""
        self.cleanup()  # Cleanup the problem
        self.model.invalidate()  # Invalidate the model
        # Invalidate the problem
        super(LEGOProblem, self).invalidate()

    @cached_property
    def case_reader_path(self):
        # type: () -> str
        """:obj:`str`: Path to the case reader that has been added to the driver."""
        filename = 'case_reader_' + self.output_case_string + '.sql'
        if self.data_folder:
            return os.path.join(self.data_folder, filename)
        else:
            return filename

    @cached_property
    def model_view_path(self):
        # type: () -> str
        """:obj:`str`: Path to the model view html view that is created with store_model_view()."""
        filename = 'n2_' + self.output_case_string + '.html'
        if self.data_folder:
            return os.path.join(self.data_folder, filename)
        else:
            return filename

    @cached_property
    def drivers(self):
        # type: () -> dict
        """:obj:`dict`: Dictionary containing the optimizer and DOE element UIDs from the CMDOWS
        file."""
        optimizer_elems = self.elem_arch_elems.findall('executableBlocks/optimizers/optimizer')
        doe_elems = self.elem_arch_elems.findall('executableBlocks/does/doe')
        optimizers = [elem.get('uID') for elem in optimizer_elems]
        does = [elem.get('uID') for elem in doe_elems]
        return {'optimizers': optimizers, 'does': does}

    @cached_property
    def driver_type(self):
        # type: () -> Union[str, None]
        """:obj:`str`: Type of driver as string."""
        if self.driver_uid:
            return self.loop_element_types[self.driver_uid]
        else:
            return None

    @cached_property
    def model(self):
        # type: () -> LEGOModel
        """:obj:`LEGOModel`: The LEGOModel that is automatically built from the CMDOWS file and
        knowledge base."""
        return LEGOModel(
            cmdows_path=self.cmdows_path,
            kb_path=self.kb_path,
            driver_uid=self.driver_uid,
            data_folder=self.data_folder,
            base_xml_file=self.base_xml_file,
            discipline_resolvers=self.discipline_resolvers.copy(),
            keep_files=self.keep_files,
        )

    @cached_property
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
        if self.driver_type == 'optimizer':
            # Find optimizer element in CMDOWS file
            opt_uid = self.driver_uid
            opt_elem = get_element_by_uid(self.elem_cmdows, opt_uid)
            # Load settings from CMDOWS file
            opt_package = get_opt_setting_safe(opt_elem, 'package', 'SciPy')
            opt_algo = get_opt_setting_safe(opt_elem, 'algorithm', 'SLSQP')
            opt_maxiter = get_opt_setting_safe(opt_elem, 'maximumIterations', 50,
                                               expected_type='int')
            opt_convtol = get_opt_setting_safe(opt_elem, 'convergenceTolerance', 1e-6,
                                               expected_type='float')

            # Apply settings to the driver
            # driver
            if opt_package == 'SciPy':
                driver = ScipyOptimizeDriver()
            elif opt_package == 'pyOptSparse':
                try:
                    from openmdao.api import pyOptSparseDriver
                except ImportError:
                    raise PyOptSparseImportError()
                driver = pyOptSparseDriver()
            else:
                raise ValueError('Unsupported package {} encountered in CMDOWS file for "{}".'
                                 .format(opt_package, opt_uid))

            # optimization algorithm
            if opt_algo == 'SLSQP':
                driver.options['optimizer'] = 'SLSQP'
            elif opt_algo == 'COBYLA':
                driver.options['optimizer'] = 'COBYLA'
            elif opt_algo == 'L-BFGS-B':
                driver.options['optimizer'] = 'L-BFGS-B'
            else:
                raise ValueError('Unsupported algorithm {} encountered in CMDOWS file for "{}".'
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
                driver.options['disp'] = False  # Print the result
            return driver
        elif self.driver_type == 'doe':
            # Find DOE element in CMDOWS file
            doe_uid = self.driver_uid
            doe_elem = get_element_by_uid(self.elem_cmdows, doe_uid)
            # Load settings from CMDOWS file
            doe_method = get_doe_setting_safe(doe_elem, 'method', 'Uniform design') # type: str
            doe_runs = get_doe_setting_safe(doe_elem, 'runs', 5, expected_type='int',
                                            doe_method=doe_method,
                                            required_for_doe_methods=['Latin hypercube design',
                                                                      'Uniform design',
                                                                      'Monte Carlo design'])
            doe_center_runs = get_doe_setting_safe(doe_elem, 'centerRuns', 2, expected_type='int',
                                                   doe_method=doe_method,
                                                   required_for_doe_methods=['Box-Behnken design'])
            doe_seed = get_doe_setting_safe(doe_elem, 'seed', None, expected_type='int',
                                            doe_method=doe_method,
                                            required_for_doe_methods=['Latin hypercube design',
                                                                      'Uniform design',
                                                                      'Monte Carlo design'])
            doe_levels = get_doe_setting_safe(doe_elem, 'levels', 2, expected_type='int',
                                              doe_method=doe_method,
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
                        value = float(row_elem.find('tableElement[@experimentID="{}"]'
                                                    .format(idx)).text)
                        data_sample.append([row_elem.attrib['relatedParameterUID'], value])
                    doe_data.append(data_sample)
            else:
                if doe_method in ['Custom design table']:
                    raise ValueError('Table element with data for custom design table missing in '
                                     'CMDOWS file.')

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
                driver.options['generator'] = LatinHypercubeGenerator(samples=doe_runs,
                                                                      criterion='maximin',
                                                                      seed=doe_seed)
            elif doe_method == 'Custom design table':
                driver.options['generator'] = ListGenerator(data=doe_data)
            else:
                raise ValueError('Could not match the doe_method {} with methods from OpenMDAO.'
                                 .format(doe_method))
            return driver
        else:
            return Driver()

    def clean_driver_after_failure(self):
        # type: () -> None
        """Clean the driver of an OpenMDAO Probem() object. This is done if the driver
           (optimization) has failed and nan (not a number) values are stored in the inputs and
           outputs."""
        for inp in self.model.list_inputs(out_stream=None):
            prom_name = self.model._var_abs2prom['input'][inp[0]]
            if np.isnan(np.min(inp[1]['value'])) or prom_name in self.model.design_vars:
                if prom_name in self.model.design_vars:
                    self[inp[0]] = np.array([self.model.design_vars[prom_name]['initial']])
                else:
                    self[inp[0]] = np.ones(len(inp[1]['value']))
        for out in self.model.list_outputs(out_stream=None):
            prom_name = self.model._var_abs2prom['output'][out[0]]
            if np.isnan(np.min(out[1]['value'])) or prom_name in self.model.design_vars:
                if prom_name in self.model.design_vars:
                    self[out[0]] = np.array([self.model.design_vars[prom_name]['initial']])
                else:
                    self[out[0]] = np.ones(len(out[1]['value']))

    def store_model_view(self, open_in_browser=False):
        # type: (bool) -> None
        """Implementation of the view_model() function for storage and (optionally) viewing in the
        browser.

        Parameters
        ----------
            open_in_browser : bool
                Setting whether to attempt to automatically open the model view in the browser.
        """
        if self._setup_status == 0:
            self.setup()
        n2(self, outfile=self.model_view_path, show_browser=open_in_browser)

    def initialize(self):
        # type: () -> None
        """Method to initialize the problem by adding a recorder and doing the setup."""
        self.driver.add_recorder(SqliteRecorder(self.case_reader_path))
        self.driver.recording_options['includes'] = ['*']
        self.driver.recording_options['record_model_metadata'] = True
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
        for xpath, value in xml_to_dict(xml).items():
            name = xpath_to_param(xpath)
            prom2abs_list_inputs = self.model._var_allprocs_prom2abs_list['input']
            prom2abs_list_outputs = self.model._var_allprocs_prom2abs_list['output']
            if self.comm.size > 1:  # Small workaround for issue in OpenMDAO with mpirun
                if name in prom2abs_list_inputs:
                    for abs_name in prom2abs_list_inputs[name]:
                        self[abs_name] = value
                if name in prom2abs_list_outputs:
                    for abs_name in prom2abs_list_outputs[name]:
                        self[abs_name] = value
            else:
                if name in prom2abs_list_inputs or name in prom2abs_list_outputs:
                    self[name] = value
            if name in self.model.mapped_parameters_inv:
                for mapping in self.model.mapped_parameters_inv[name]:
                    if mapping in prom2abs_list_inputs or mapping in prom2abs_list_outputs:
                        try:
                            # Small workaround for issue in OpenMDAO with mpirun
                            if self.comm.size > 1:
                                abs_names = []
                                if mapping in prom2abs_list_inputs:
                                    abs_names.extend([abs_name for abs_name in
                                                      prom2abs_list_inputs[mapping]])
                                if mapping in prom2abs_list_outputs:
                                    abs_names.extend([abs_name for abs_name in
                                                      prom2abs_list_outputs[mapping]])
                                for abs_name in abs_names:
                                    self[abs_name] = value
                            else:
                                self[mapping] = value
                        except RuntimeError as e:
                            if 'The promoted name' in e[0] and 'is invalid' in e[0]:
                                warnings.warn('Could not automatically set this invalid promoted '
                                              'name from the XML: {}.'.format(mapping))
                            else:
                                raise RuntimeError(e)

    def postprocess_experiments(self, vector, vector_name, failed_experiments=(None, None)):
        # type: (np.array, str, Optional[Tuple]) -> np.array
        """
        Postprocess experiments from a DOE to remove failed experiments from the vector.

        Parameters
        ----------
            vector : vector with experiment results
            vector_name : name of vector
            failed_experiments : failed experiments of other vectors (to speed up)

        Returns
        -------
            tuple with post-processed vector and failed experiments

        Raises
        ------
            AssertionError : if vector name is not found in sample lists
        """
        # Determine whether it concerns input or output sample lists
        if vector_name in [xpath_to_param(xpath) for xpath in self.doe_sample_lists['inputs']]:
            # Assert that the failed_experiments are known, or else throw an error
            if failed_experiments[0] is None:
                raise IOError('For DOE input sample lists the failed experiments need to be known '
                              'before postprocessing.')
            return np.delete(vector, list(failed_experiments[0])), failed_experiments
        elif vector_name in [xpath_to_param(xpath) for xpath in self.doe_sample_lists['outputs']]:
            # Determine the failed experiments in the vector
            vector_failures = set(np.where(np.isnan(vector))[0])

            # Add or compare failed experiments w.r.t. failed_experiments input
            if failed_experiments[0] is None:
                failed_experiments = (vector_failures, len(vector_failures) / len(vector))
            else:
                if not vector_failures == failed_experiments[0]:
                    raise AssertionError('The failed experiments of {} are not consistent in the '
                                         'training data.'
                                         .format(vector_name))
            return np.array(list(filter(lambda x: not np.isnan(x), vector))), failed_experiments
        else:
            raise AssertionError('Could not determine the vector type for vector_name: {}.'
                                 .format(vector_name))

    def get_case_reader(self):
        # type: () -> CaseReader
        """Get the case reader of the problem

        Returns
        -------
            CaseReader
                The configured case reader for the Problem object

        Raises
        ------
            AssertionError
                If case reader file cannot be found
        """
        if not os.path.isfile(self.case_reader_path):
            raise AssertionError('Could not find the case reader file {}.'
                                 .format(self.case_reader_path))
            # Get all cases from the case reader and determine amount of cases
        return CaseReader(self.case_reader_path)

    def collect_results(self, cases_to_collect='default', print_in_log=True):
        # type: (Union[str, list]) -> Dict[dict]
        """Print the results that were stored in the case reader.

        Parameters
        ----------
            cases_to_collect : str or list
                Setting on which cases should be print (e.g. 'last', 'all', 'default', [2, 3, 5])

            print_in_log : bool
                Setting on whether the results should also be printed in the log

        Returns
        -------
            results : dict of dicts
                Dictionary containing the results that were collected
        """
        if not isinstance(cases_to_collect, (str, list)):
            raise AssertionError('cases_to_print must be of type str or list.')
        if isinstance(cases_to_collect, str):
            if cases_to_collect not in ['default', 'all', 'last']:
                raise AssertionError('Invalid cases_to_print string value provided.')
        if cases_to_collect == 'default':
            if self.driver_type == 'doe':
                cases_to_collect = 'all'
            elif self.driver_type == 'optimizer':
                cases_to_collect = 'last'
            else:
                cases_to_collect = 'last'
        results = dict()

        # Get all cases from the case reader and determine amount of cases
        cr = self.get_case_reader()
        cases = cr.list_cases('driver')
        num_cases = len(cases)

        if num_cases == 0:
            raise AssertionError('No cases were recorded and therefore no results can be collected.'
                                 ' Note that collect_results only works after the driver has been '
                                 'run.')

        # Change cases_to_print to a list of integers with case numbers
        if isinstance(cases_to_collect, str):
            if cases_to_collect == 'all':
                cases_to_collect = range(num_cases)
            elif cases_to_collect == 'last':
                cases_to_collect = [num_cases - 1]

        # Print results
        print_optional('\nPrinting results from case reader: {}.'.format(self.case_reader_path),
                       print_in_log)
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
            case = cr.get_case(cases[num_case])
            print_optional('\n\n  Case {}/{} ({})'.format(num_case, num_cases-1,
                                                          case.iteration_coordinate),
                           print_in_log)

            # Get objectives, design variables and contraints
            recorded_objectives = case.get_objectives(scaled=False)
            recorded_design_vars = case.get_design_vars(scaled=False)
            recorded_constraints = case.get_constraints(scaled=False)

            # Get objectives, design variables and contraints
            var_objectives = sorted(list(recorded_objectives.keys()))
            var_design_vars = sorted(list(recorded_design_vars.keys()))
            var_constraints = sorted(list(recorded_constraints.keys()))

            var_does = []
            if isinstance(self.driver, DOEDriver):
                var_does = sorted([elem.text for elem in self.elem_arch_elems
                                  .findall('parameters/doeOutputSampleLists/doeOutputSampleList/'
                                           'relatedParameterUID')])
            var_convs = sorted([elem.text for elem in self.elem_problem_def
                               .findall('problemRoles/parameters/stateVariables/stateVariable/'
                                        'parameterUID')])

            # Print objective
            if var_objectives:
                print_optional('    Objectives', print_in_log)
                for var_objective in var_objectives:
                    value = recorded_objectives[xpath_to_param(var_objective)]
                    print_optional('    {}: {}'.format(var_objective, value), print_in_log)
                    results = add_or_append_dict_entry(results, 'objectives', var_objective, value)

            # Print design variables
            if var_design_vars:
                print_optional('\n    Design variables', print_in_log)
                for var_desvar in var_design_vars:
                    metadata_name = cr._prom2abs['output'][var_desvar][0]
                    value = recorded_design_vars[var_desvar]
                    if len(value) == 1:
                        value = value[0]
                    lb_value = cr.problem_metadata['variables'][metadata_name]['ref0']
                    ub_value = cr.problem_metadata['variables'][metadata_name]['ref']
                    print_optional('    {}: {} ({} < x < {})'.format(var_desvar, value, lb_value,
                                                                     ub_value), print_in_log)
                    results = add_or_append_dict_entry(results, 'desvars', var_desvar, value)

            # Print constraint values
            if var_constraints:
                print_optional('\n    Constraints', print_in_log)
                for var_constraint in var_constraints:
                    metadata_name = cr._prom2abs['output'][var_constraint][0]
                    value = recorded_constraints[var_constraint]
                    if len(value) == 1:
                        value = value[0]
                    lb_value = cr.problem_metadata['variables'][metadata_name]['lower']
                    ub_value = cr.problem_metadata['variables'][metadata_name]['upper']
                    eq_value = cr.problem_metadata['variables'][metadata_name]['equals']
                    if eq_value is not None:
                        print_optional('    {}: {} (c == {})'.format(var_constraint, value,
                                                                     eq_value), print_in_log)
                    else:
                        if lb_value > -1e29 and ub_value < 1e29:
                            print_optional('    {}: {} ({} < c < {})'
                                           .format(var_constraint, value, lb_value, ub_value),
                                           print_in_log)
                        elif lb_value < -1e29 and ub_value < 1e29:
                            print_optional('    {}: {} (c < {})'.format(var_constraint, value,
                                                                        ub_value), print_in_log)
                        elif lb_value > -1e29 and ub_value > 1e29:
                            print_optional('    {}: {} (c > {})'.format(var_constraint, value,
                                                                        lb_value), print_in_log)
                        else:
                            print_optional('    {}: {} (c is unbounded)'
                                           .format(var_constraint, value), print_in_log)
                    results = add_or_append_dict_entry(results, 'constraints',
                                                       var_constraint, value)

            # Print DOE quantities of interest
            if var_does:
                print_optional('\n    Quantities of interest', print_in_log)
                for var_qoi in var_does:
                    if var_qoi in var_does:
                        value = case.outputs[var_qoi]
                        if len(value) == 1:
                            value = value[0]
                        print_optional('    {}: {}'.format(var_qoi, value), print_in_log)
                        results = add_or_append_dict_entry(results, 'qois', var_qoi, value)

            # Print other quantities of interest
            title_not_printed = True
            if var_convs:
                for var_qoi in var_convs:
                    if var_qoi not in var_objectives + var_constraints + var_does:
                        if title_not_printed:
                            print_optional('\n    Quantities of interest', print_in_log)
                            title_not_printed = False
                        value = case.outputs[var_qoi]
                        if len(value) == 1:
                            value = value[0]
                        print_optional('    {}: {}'.format(var_qoi, value), print_in_log)
                        results = add_or_append_dict_entry(results, 'qois', var_qoi, value)
        return results
