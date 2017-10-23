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

This file contains the test case for the wing optimization example problem.
"""
from __future__ import absolute_import, division, print_function

import unittest

from openmdao.api import Problem, ScipyOptimizer

from openlego.api import LEGOModel
from openlego.recorders import NormalizedDesignVarPlotter, ConstraintsPlotter, SimpleObjectivePlotter


class TestWingOptimization(unittest.TestCase):

    def test_wing_opt(self):
        """Solve the wing optimization problem."""
        # 1. Create a Problem
        prob = Problem()
        prob.set_solver_print(0)

        # 2. Create the LEGOModel
        model = prob.model = LEGOModel('wing_opt_MDG_MDF_GS.xml',           # CMDOWS file
                                       'kb',                                # Knowledge base
                                       '',                                  # Output directory
                                       'wing_opt_output.xml')               # Output file

        # 3. Create a Driver object
        driver = prob.driver = ScipyOptimizer()
        driver.options['optimizer'] = 'SLSQP'
        driver.options['disp'] = True
        driver.options['tol'] = 1.0e-3
        driver.opt_settings = {'disp': True, 'iprint': 2, 'ftol': 1.0e-3}

        # 4. Setup the problem
        prob.setup()
        prob.run_model()
        model.initialize_from_xml('wing_opt_input.xml')

        # 5. Attach some Recorders
        driver.add_recorder(NormalizedDesignVarPlotter())
        driver.add_recorder(ConstraintsPlotter())
        driver.add_recorder(SimpleObjectivePlotter())

        # 6. Solve the problem
        prob.run_driver()
        prob.cleanup()
