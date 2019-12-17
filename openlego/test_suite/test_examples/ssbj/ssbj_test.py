#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2019 I. van Gent

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains the definition of the Super-Sonic Business Jet test cases.
"""
from __future__ import absolute_import, division, print_function

import os
import logging
import unittest

from typing import Union, Optional, List

from openlego.core.problem import LEGOProblem
import openlego.test_suite.test_examples.ssbj.kb as kb
from openlego.utils.general_utils import clean_dir_filtered, pyoptsparse_installed

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

# List of MDAO definitions that can be wrapped around the problem
mdao_definitions = ['unconverged-MDA-GS',     # 0
                    'unconverged-MDA-J',      # 1
                    'converged-MDA-GS',       # 2
                    'converged-MDA-J',        # 3
                    'unconverged-DOE-GS-CT',  # 4
                    'unconverged-DOE-J-CT',   # 5
                    'converged-DOE-GS-CT',    # 6
                    'converged-DOE-J-CT',     # 7
                    'converged-DOE-GS-LH',    # 8
                    'converged-DOE-GS-MC',    # 9
                    'MDF-GS',                 # 10
                    'MDF-J',                  # 11
                    'IDF',                    # 12
                    'CO',                     # 13
                    'BLISS-2000']             # 14


def get_loop_items(analyze_mdao_definitions):
    # type: (Union[int, list, str]) -> List[str]
    """Retrieve the list of MDAO definitions to be analyzed based on different input settings.

    Parameters
    ----------
        analyze_mdao_definitions : Union[int, list, str]
            Indicator for the definitions to be analyzed. Can be an int, a list, or the string 'all'

    Returns
    -------
        mdao_defs_loop : List[str]
            List containing the MDAO definition to be analyzed
    """
    if isinstance(analyze_mdao_definitions, int):
        mdao_defs_loop = [mdao_definitions[analyze_mdao_definitions]]
    elif isinstance(analyze_mdao_definitions, list):
        mdao_defs_loop = [mdao_definitions[i] for i in analyze_mdao_definitions]
    elif isinstance(analyze_mdao_definitions, str):
        if analyze_mdao_definitions == 'all':
            mdao_defs_loop = mdao_definitions
        else:
            raise ValueError('String value {} is not allowed for analyze_mdao_definitions.'
                             .format(analyze_mdao_definitions))
    else:
        raise IOError('Invalid input {} provided of type {}.'
                      .format(analyze_mdao_definitions, type(analyze_mdao_definitions)))
    return mdao_defs_loop


def run_openlego(analyze_mdao_definitions, cmdows_dir=None, initial_file_path=None,
                 data_folder=None, run_type='test', approx_totals=False, driver_debug_print=False):
    # type: (Union[int, list, str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[bool], Optional[bool]) -> Union[tuple, LEGOProblem]
    """Run OpenLEGO for a list of MDAO definitions.

    Parameters
    ----------
    analyze_mdao_definitions : list
        List of MDAO definitions to be analyzed.

    cmdows_dir : str
        Path to directory with CMDOWS files

    initial_file_path : str
        Path to file containing initial values

    data_folder : str
        Path to directory where results will be stored

    run_type : str
        Option to indicate the type of run, as this changes the return statement used

    approx_totals : bool
        Setting on whether to use approx_totals on the model

    driver_debug_print : bool
        Setting on whether to print debug information in the log

    Returns
    -------
        Union[Tuple[float], LEGOProblem]
    """
    # Check and analyze inputs
    mdao_defs_loop = get_loop_items(analyze_mdao_definitions)
    file_dir = os.path.dirname(__file__)
    if not cmdows_dir:
        cmdows_dir = os.path.join(file_dir, 'cmdows_files')
    if not initial_file_path:
        initial_file_path = os.path.join(file_dir, 'SSBJ-base.xml')
    if not data_folder:
        data_folder = ''

    # Run the
    for mdao_def in mdao_defs_loop:
        print('\n-----------------------------------------------')
        print('Running the OpenLEGO of Mdao_{}.xml...'.format(mdao_def))
        print('------------------------------------------------')
        """Solve the SSBJ problem using the given CMDOWS file."""

        # 1. Create Problem
        prob = LEGOProblem(cmdows_path=os.path.join(cmdows_dir, 'Mdao_{}.xml'.format(mdao_def)),
                           kb_path=os.path.join(file_dir, 'kb'),  # Knowledge base path
                           data_folder=data_folder,  # Output directory
                           base_xml_file=os.path.join(data_folder,
                                                      'ssbj-output-{}.xml'.format(mdao_def)))
        if driver_debug_print:
            prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'ln_cons', 'objs']
        prob.set_solver_print(0)  # Set printing of solver information

        if approx_totals:
            prob.model.approx_totals()

        # 2. Initialize the Problem and export N2 chart
        prob.store_model_view()
        prob.initialize_from_xml(initial_file_path)  # Set the initial values from an XML file

        # 3. Run the Problem
        test_distributed = mdao_def in ['CO', 'BLISS-2000'] and run_type == 'test'
        if test_distributed:
            prob.run_model()
        else:
            prob.run_driver()  # Run the driver (optimization, DOE, or convergence)

        # 4. Read out the case reader
        if not test_distributed:
            prob.collect_results()

        if run_type == 'test':
            # 5. Collect test results for test assertions
            tc = prob['/dataSchema/aircraft/geometry/tc'][0]
            h = prob['/dataSchema/reference/h'][0]
            M = prob['/dataSchema/reference/M'][0]
            AR = prob['/dataSchema/aircraft/geometry/AR'][0]
            Lambda = prob['/dataSchema/aircraft/geometry/Lambda'][0]
            Sref = prob['/dataSchema/aircraft/geometry/Sref'][0]
            if mdao_def not in ['CO', 'BLISS-2000']:
                lambda_ = prob['/dataSchema/aircraft/geometry/lambda'][0]
                section = prob['/dataSchema/aircraft/geometry/section'][0]
                Cf = prob['/dataSchema/aircraft/other/Cf'][0]
                T = prob['/dataSchema/aircraft/other/T'][0]
                R = prob['/dataSchema/scaledData/R/value'][0]
                extra = prob['/dataSchema/aircraft/weight/WT'][0]
            elif mdao_def == 'CO':
                lambda_ = prob.model.SubOptimizer0.prob['/dataSchema/aircraft/geometry/lambda'][0]
                section = prob.model.SubOptimizer0.prob['/dataSchema/aircraft/geometry/section'][0]
                Cf = prob.model.SubOptimizer1.prob['/dataSchema/aircraft/other/Cf'][0]
                T = prob.model.SubOptimizer2.prob['/dataSchema/aircraft/other/T'][0]
                R = prob['/dataSchema/scaledData/R/value'][0]
                extra = (prob['/dataSchema/distributedArchitectures/group0/objective'],
                         prob['/dataSchema/distributedArchitectures/group1/objective'],
                         prob['/dataSchema/distributedArchitectures/group2/objective'])
            else:
                lambda_, section, Cf, T, R, extra = None, None, None, None, None, None

            # 6. Cleanup and invalidate the Problem afterwards
            prob.invalidate()
            return tc, h, M, AR, Lambda, Sref, lambda_, section, Cf, T, R, extra
        elif run_type == 'validation':
            return prob
        else:
            prob.invalidate()


class TestSsbj(unittest.TestCase):
    """Test class to run the SSBJ test case for a range of architectures.
    """

    def assertion_mda(self, tc, h, M, AR, Lambda, Sref, lambda_, section, Cf, T, R, extra):
        self.assertAlmostEqual(tc, .05, 2)
        self.assertAlmostEqual(h, 45000., 2)
        self.assertAlmostEqual(M, 1.6, 2)
        self.assertAlmostEqual(AR, 5.5, 2)
        self.assertAlmostEqual(Lambda, 55., 2)
        self.assertAlmostEqual(Sref, 1000., 2)
        self.assertAlmostEqual(lambda_, .25, 2)
        self.assertAlmostEqual(section, 1., 2)
        self.assertAlmostEqual(Cf, 1., 2)
        self.assertAlmostEqual(T, .2, 2)
        self.assertAlmostEqual(R, -.7855926, 2)
        self.assertAlmostEqual(extra, 63609.56, delta=1.)

    def assertion_unc_doe_gs(self, tc, h, M, AR, Lambda, Sref, lambda_, section, Cf, T, R, extra):
        self.assertAlmostEqual(tc, .09, 2)
        self.assertAlmostEqual(h, 60000., 2)
        self.assertAlmostEqual(M, 1.8, 2)
        self.assertAlmostEqual(AR, 8.5, 2)
        self.assertAlmostEqual(Lambda, 70., 2)
        self.assertAlmostEqual(Sref, 1500., 2)
        self.assertAlmostEqual(lambda_, .4, 2)
        self.assertAlmostEqual(section, 1.25, 2)
        self.assertAlmostEqual(Cf, 1.25, 2)
        self.assertAlmostEqual(T, 1., 2)
        self.assertAlmostEqual(R, -1.15528254, 2)
        self.assertAlmostEqual(extra, 149272.43, delta=1.)

    def assertion_unc_doe_j(self, tc, h, M, AR, Lambda, Sref, lambda_, section, Cf, T, R, extra):
        self.assertAlmostEqual(tc, .09, 2)
        self.assertAlmostEqual(h, 60000., 2)
        self.assertAlmostEqual(M, 1.8, 2)
        self.assertAlmostEqual(AR, 8.5, 2)
        self.assertAlmostEqual(Lambda, 70., 2)
        self.assertAlmostEqual(Sref, 1500., 2)
        self.assertAlmostEqual(lambda_, .4, 2)
        self.assertAlmostEqual(section, 1.25, 2)
        self.assertAlmostEqual(Cf, 1.25, 2)
        self.assertAlmostEqual(T, 1., 2)
        self.assertAlmostEqual(R, -0.82379969, 2)
        self.assertAlmostEqual(extra, 148199.57, delta=1.)

    def assertion_con_doe(self, tc, h, M, AR, Lambda, Sref, lambda_, section, Cf, T, R, extra):
        self.assertAlmostEqual(tc, .09, 2)
        self.assertAlmostEqual(h, 60000., 2)
        self.assertAlmostEqual(M, 1.8, 2)
        self.assertAlmostEqual(AR, 8.5, 2)
        self.assertAlmostEqual(Lambda, 70., 2)
        self.assertAlmostEqual(Sref, 1500., 2)
        self.assertAlmostEqual(lambda_, .4, 2)
        self.assertAlmostEqual(section, 1.25, 2)
        self.assertAlmostEqual(Cf, 1.25, 2)
        self.assertAlmostEqual(T, 1., 2)
        self.assertAlmostEqual(R, -1.13518441, 2)
        self.assertAlmostEqual(extra, 187899.83, delta=1.)

    def assertion_mdo(self, tc, h, M, AR, Lambda, Sref, lambda_, section, Cf, T, R, extra):
        self.assertAlmostEqual(tc, .06, 2)
        self.assertAlmostEqual(h, 60000., 2)
        self.assertAlmostEqual(M, 1.4, 2)
        self.assertAlmostEqual(AR, 2.475, 2)
        self.assertAlmostEqual(Lambda, 69.85, 2)
        self.assertAlmostEqual(Sref, 1500., 2)
        self.assertAlmostEqual(lambda_, .4, 2)
        self.assertAlmostEqual(section, .75, 2)
        self.assertAlmostEqual(Cf, .75, 2)
        self.assertAlmostEqual(T, .15620845, 2)
        self.assertAlmostEqual(R, -7.40624897, 2)
        self.assertAlmostEqual(extra, 44957.70, delta=100.)

    def assertion_co(self, tc, h, M, AR, Lambda, Sref, lambda_, section, Cf, T, R, extra):
        self.assertAlmostEqual(tc, .05, 2)
        self.assertAlmostEqual(h, 45000., 2)
        self.assertAlmostEqual(M, 1.6, 2)
        self.assertAlmostEqual(AR, 5.5, 2)
        self.assertAlmostEqual(Lambda, 55., 2)
        self.assertAlmostEqual(Sref, 1000., 2)
        self.assertAlmostEqual(lambda_, .15, 2)
        self.assertAlmostEqual(section, 1., 2)
        self.assertAlmostEqual(Cf, 1., 2)
        self.assertAlmostEqual(T, .2, 2)
        self.assertAlmostEqual(R, -.7855926, 2)
        for J in extra:
            self.assertAlmostEqual(J, 0., delta=0.1)

    def assertion_b2k(self, tc, h, M, AR, Lambda, Sref, lambda_, section, Cf, T, R, extra):
        self.assertTrue(isinstance(tc, float))
        self.assertTrue(isinstance(h, float))
        self.assertTrue(isinstance(M, float))
        self.assertTrue(isinstance(AR, float))
        self.assertTrue(isinstance(Lambda, float))
        self.assertTrue(isinstance(Sref, float))
        self.assertTrue(lambda_ is None)
        self.assertTrue(section is None)
        self.assertTrue(Cf is None)
        self.assertTrue(T is None)
        self.assertTrue(R is None)
        self.assertTrue(extra is None)

    def test_unc_mda_gs(self):
        """Test run the SSBJ tools in sequence."""
        self.assertion_mda(*run_openlego(0))

    def test_unc_mda_j(self):
        """Test run the SSBJ tools in parallel."""
        self.assertion_mda(*run_openlego(1))

    def test_con_mda_gs(self):
        """Solve the SSBJ system using the Gauss-Seidel convergence scheme."""
        self.assertion_mda(*run_openlego(2))

    def test_con_mda_j(self):
        """Solve the SSBJ system using the Jacobi convergence scheme."""
        self.assertion_mda(*run_openlego(3))

    def test_unc_doe_gs_ct(self):
        """Solve multiple (DOE) SSBJ systems (unconverged) in sequence based on a custom design table."""
        self.assertion_unc_doe_gs(*run_openlego(4))

    def test_unc_doe_j_ct(self):
        """Solve multiple (DOE) SSBJ systems (unconverged) in parallel based on a custom design table."""
        self.assertion_unc_doe_j(*run_openlego(5))

    def test_con_doe_gs_ct(self):
        """Solve multiple (DOE) SSBJ systems (converged) in sequence based on a custom design table."""
        self.assertion_con_doe(*run_openlego(6))

    def test_con_doe_j_ct(self):
        """Solve multiple (DOE) SSBJ systems (converged) in parallel based on a custom design table."""
        self.assertion_con_doe(*run_openlego(7))

    def test_con_doe_gs_lh(self):
        """Solve multiple (DOE) SSBJ systems (converged) in sequence based on a latin hypercube sampling."""
        run_openlego(8)  # Note: this test is not asserted as the DOE choose a random experiments

    def test_con_doe_gs_mc(self):
        """Solve multiple (DOE) SSBJ systems (converged) in sequence based on a Monte Carlo sampling."""
        run_openlego(9)  # Note: this test is not asserted as the DOE chooses random experiments

    def test_mdf_gs(self):
        """Solve the SSBJ problem using the MDF architecture and a Gauss-Seidel convergence scheme."""
        self.assertion_mdo(*run_openlego(10))

    def test_mdf_j(self):
        """Solve the SSBJ problem using the MDF architecture and a Jacobi converger."""
        self.assertion_mdo(*run_openlego(11))

    def test_idf(self):
        """Solve the SSBJ problem using the IDF architecture."""
        self.assertion_mdo(*run_openlego(12))

    def test_co(self):
        """Test run the SSBJ problem using the CO architecture."""
        if pyoptsparse_installed():
            self.assertion_co(*run_openlego(13))
        else:
            print('\nSkipped test due to missing PyOptSparse installation.')
            pass

    def test_b2k(self):
        """Test run the SSBJ problem using the BLISS-2000 architecture."""
        if pyoptsparse_installed():
            self.assertion_b2k(*run_openlego(14))
        else:
            print('\nSkipped test due to missing PyOptSparse installation.')
            pass

    def __del__(self):
        kb.clean()
        clean_dir_filtered(os.path.dirname(__file__), ['case_reader_', 'n2_Mdao_', 'ssbj-output-',
                                                       'SLSQP.out', 'b2k_'])


if __name__ == '__main__':
    unittest.main()
