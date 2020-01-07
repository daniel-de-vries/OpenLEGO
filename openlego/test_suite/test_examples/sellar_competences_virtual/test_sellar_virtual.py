#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2019 D. de Vries and I. van Gent

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains the definition of the Sellar with mathematical relations test cases.
"""
from __future__ import absolute_import, division, print_function

import logging
import os
import unittest
import tempfile

from openlego.core.problem import LEGOProblem
from openlego.test_suite.test_examples.sellar_functions import get_couplings, get_objective, get_g1, \
    get_g2
from openlego.utils.general_utils import clean_dir_filtered
from openlego.core.discipline_resolver import DisciplineInstanceResolver
from openlego.core.abstract_discipline import AbstractDiscipline
from openlego.test_suite.test_examples.sellar_competences_virtual.disciplines import *

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def run_openlego(cmdows_file, data_folder=None, driver_debug_print=False, discipline_resolvers=None):
    file_dir = os.path.dirname(__file__)
    initial_file_path = os.path.join(file_dir, 'input.xml')
    if not data_folder:
        data_folder = ''

    print('\n-----------------------------------------------')
    print('Running the OpenLEGO of {}...'.format(cmdows_file))
    print('------------------------------------------------')
    # 1. Create Problem
    prob = LEGOProblem(
        cmdows_path=os.path.join(file_dir, 'cmdows', cmdows_file),
        kb_path='',
        data_folder=data_folder,
        base_xml_file='sellar-output.xml',
        discipline_resolvers=discipline_resolvers,
    )
    if driver_debug_print:
        prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'ln_cons', 'objs']
    prob.set_solver_print(0)  # Set printing of solver information

    # 2. Initialize the Problem and export N2 chart
    prob.store_model_view(open_in_browser=False)
    prob.initialize_from_xml(initial_file_path)

    # 3. Run the Problem
    prob.run_driver()  # Run the driver (optimization, DOE, or convergence)

    # 4. Read out the case reader
    prob.collect_results()

    # 5. Collect test results for test assertions
    x = [prob['/dataSchema/variables/x1']]
    y = [prob['/dataSchema/analyses/y1'], prob['/dataSchema/analyses/y2']]
    z = [prob['/dataSchema/variables/z1'], prob['/dataSchema/variables/z2']]
    f = [prob['/dataSchema/output/f']]
    g = [prob['/dataSchema/output/g1'], prob['/dataSchema/output/g2']]

    # 6. Cleanup and invalidate the Problem afterwards
    prob.invalidate()
    return x, y, z, f, g


class TestSellarMath(unittest.TestCase):

    def assertion_con_mda(self, x, y, z, f, g):
        self.assertAlmostEqual(float(x[0]), 4.00, 2)
        self.assertAlmostEqual(float(y[0]), float(get_couplings(z[0], z[1], x[0])[0]), 2)
        self.assertAlmostEqual(float(y[1]), float(get_couplings(z[0], z[1], x[0])[1]), 2)
        self.assertAlmostEqual(float(z[0]), 1.00, 2)
        self.assertAlmostEqual(float(z[1]), 5.00, 2)
        self.assertAlmostEqual(float(f[0]), get_objective(x[0], z[1], y[0], y[1]), 2)
        self.assertAlmostEqual(float(g[0]), get_g1(y[0]), 2)
        self.assertAlmostEqual(float(g[1]), get_g2(y[1]), 2)

    def assertion_mdo(self, x, y, z, f, g):
        self.assertAlmostEqual(float(x[0]), 0.00, delta=.1)
        self.assertAlmostEqual(float(y[0]), get_couplings(z[0], z[1], x[0])[0], delta=0.1)
        self.assertAlmostEqual(float(y[1]), get_couplings(z[0], z[1], x[0])[1], delta=0.1)
        self.assertAlmostEqual(float(z[0]), 1.98, delta=.1)
        self.assertAlmostEqual(float(z[1]), 0.00, delta=.1)
        self.assertAlmostEqual(float(f[0]), 3.18, delta=.1)
        self.assertAlmostEqual(float(g[0]), 0.00, delta=.1)
        self.assertAlmostEqual(float(g[1]), 0.84, delta=.1)

    @staticmethod
    def virtual_kb():
        return DisciplineInstanceResolver([
            D1Discipline(),
            D2Discipline(),
            F1Discipline(),
            G1Discipline(),
            G2Discipline(),
        ])

    def test_virtual_kb(self):
        virtual_kb = self.virtual_kb()
        self.assertEqual(len(virtual_kb.disciplines), 5)

        for discipline in virtual_kb.disciplines:
            self.assertIsInstance(discipline, AbstractDiscipline)

            name = discipline.name
            resolved = virtual_kb.resolve_discipline(name, 'main')
            self.assertIs(discipline, resolved)

        self.assertIsNone(virtual_kb.resolve_discipline('NonExistingDiscipline', 'main'))
        self.assertIsNone(virtual_kb.resolve_discipline('D1', 'special_mode'))

    def test_write_io(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            virtual_kb = self.virtual_kb()
            virtual_kb.write_io(tmp_dir)

            for discipline in virtual_kb.disciplines:
                assert os.path.exists(os.path.join(tmp_dir, discipline.name+'-input.xml'))
                assert os.path.exists(os.path.join(tmp_dir, discipline.name+'-output.xml'))

    def test_mda_gs(self):
        self.assertion_con_mda(*run_openlego('cmdows_mdax_Sellar_Test_MDA-GS.xml',
                                             discipline_resolvers=[self.virtual_kb()]))

    def test_mdf_gs(self):
        self.assertion_mdo(*run_openlego('cmdows_mdax_Sellar_Test_MDF-GS.xml',
                                         discipline_resolvers=[self.virtual_kb()]))

    def test_mda_gs_self_loop(self):
        virtual_kb = DisciplineInstanceResolver([
            D12Discipline(),
            F1Discipline(),
            G1Discipline(),
            G2Discipline(),
        ])
        self.assertion_con_mda(*run_openlego('cmdows_mdax_Sellar_Test_MDA-GS-self.xml',
                                             discipline_resolvers=[virtual_kb]))

    def __del__(self):
        clean_dir_filtered(os.path.dirname(__file__), ['case_reader_', 'n2_cmdows_',
                                                       'sellar-output.xml', 'SLSQP.out'])


if __name__ == '__main__':
    unittest.main()
