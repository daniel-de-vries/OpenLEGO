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

This file contains the definition of the Super-Sonic Business Jet test cases.
"""
from __future__ import absolute_import, division, print_function

import os
import logging
import unittest

from openlego.core.problem import LEGOProblem
import openlego.test_suite.test_examples.ssbj.kb as kb
from openlego.utils.general_utils import clean_dir_filtered

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
    if isinstance(analyze_mdao_definitions, int):
        mdao_defs_loop = [mdao_definitions[analyze_mdao_definitions]]
    elif isinstance(analyze_mdao_definitions, list):
        mdao_defs_loop = [mdao_definitions[i] for i in analyze_mdao_definitions]
    elif isinstance(analyze_mdao_definitions, str):
        if analyze_mdao_definitions == 'all':
            mdao_defs_loop = mdao_definitions
        else:
            raise ValueError(
                'String value {} is not allowed for analyze_mdao_definitions.'.format(analyze_mdao_definitions))
    else:
        raise IOError(
            'Invalid input {} provided of type {}.'.format(analyze_mdao_definitions, type(analyze_mdao_definitions)))
    return mdao_defs_loop


def run_openlego(analyze_mdao_definitions):
    # Check and analyze inputs
    mdao_defs_loop = get_loop_items(analyze_mdao_definitions)

    for mdao_def in mdao_defs_loop:
        print('\n-----------------------------------------------')
        print('Running the OpenLEGO of Mdao_{}.xml...'.format(mdao_def))
        print('------------------------------------------------')
        """Solve the SSBJ problem using the given CMDOWS file."""

        # 1. Create Problem
        prob = LEGOProblem(os.path.join('cmdows_files', 'Mdao_{}.xml'.format(mdao_def)),  # CMDOWS file
                                        'kb',  # Knowledge base path
                                        '',  # Output directory
                                        'ssbj-output-{}.xml'.format(mdao_def))  # Output file
        # prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'ln_cons', 'objs']  # Set printing of debug info
        prob.set_solver_print(0)  # Set printing of solver information

        # 2. Initialize the Problem and export N2 chart
        prob.store_model_view()
        if mdao_def in ['MDF-GS', 'MDF-J', 'IDF', 'CO', 'BLISS-2000']:
            prob.initialize_from_xml('SSBJ-base-mdo.xml')  # Set the initial values from an XML file
        else:
            prob.initialize_from_xml('SSBJ-base-mda.xml')

        # 3. Run the Problem
        prob.run_driver()  # Run the driver (optimization, DOE, or convergence)

        # 4. Read out the case reader
        prob.collect_results()

        # 5. Cleanup and invalidate the Problem afterwards
        prob.invalidate()


class TestSsbj(unittest.TestCase):

    def __call__(self, *args, **kwargs):
        kb.deploy()
        super(TestSsbj, self).__call__(*args, **kwargs)

    def test_unc_mda_gs(self):
        """Test run the SSBJ tools in sequence."""
        run_openlego(0)

    def test_unc_mda_j(self):
        """Test run the SSBJ tools in parallel."""
        run_openlego(1)

    def test_con_mda_gs(self):
        """Solve the SSBJ system using the Gauss-Seidel convergence scheme."""
        run_openlego(2)

    def test_con_mda_j(self):
        """Solve the SSBJ system using the Jacobi convergence scheme."""
        run_openlego(3)

    def test_unc_doe_gs_ct(self):
        """Solve multiple (DOE) SSBJ systems (unconverged) in sequence based on a custom design table."""
        run_openlego(4)

    def test_unc_doe_j_ct(self):
        """Solve multiple (DOE) SSBJ systems (unconverged) in parallel based on a custom design table."""
        run_openlego(5)

    def test_con_doe_gs_ct(self):
        """Solve multiple (DOE) SSBJ systems (converged) in sequence based on a custom design table."""
        run_openlego(6)

    def test_con_doe_j_ct(self):
        """Solve multiple (DOE) SSBJ systems (converged) in parallel based on a custom design table."""
        run_openlego(7)

    def test_con_doe_gs_lh(self):
        """Solve multiple (DOE) SSBJ systems (converged) in sequence based on a latin hypercube sampling."""
        run_openlego(8)

    def test_con_doe_gs_mc(self):
        """Solve multiple (DOE) SSBJ systems (converged) in sequence based on a Monte Carlo sampling."""
        run_openlego(9)

    def test_mdf_gs(self):
        """Solve the SSBJ problem using the MDF architecture and a Gauss-Seidel convergence scheme."""
        run_openlego(10)

    def test_mdf_j(self):
        """Solve the SSBJ problem using the MDF architecture and a Jacobi converger."""
        run_openlego(11)

    def test_idf(self):
        """Solve the SSBJ problem using the IDF architecture."""
        run_openlego(12)

    def __del__(self):
        kb.clean()
        clean_dir_filtered(os.path.dirname(__file__), ['case_reader_', 'n2_Mdao_', 'ssbj-output-'])


if __name__ == '__main__':
    unittest.main()
