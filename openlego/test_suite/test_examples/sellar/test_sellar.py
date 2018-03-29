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

This file contains the test case for the Sellar example problem.
"""
from __future__ import absolute_import, division, print_function

import unittest

from openmdao.api import Problem, ScipyOptimizer

from openlego.api import LEGOModel


def solve_sellar(cmdows_file):
    """Solve the Sellar problem using the given CMDOWS file."""
    # 1. Create Problem
    prob = Problem()                                                # Create an instance of the Problem class
    prob.set_solver_print(0)                                        # Turn off printing of solver information

    # 2. Create the LEGOModel
    model = prob.model = LEGOModel(cmdows_file,                     # CMDOWS file
                                   'kb',                            # Knowledge base path
                                   '',                              # Output directory
                                   'sellar-output.xml')             # Output file

    # 3. Create the Driver
    driver = prob.driver = ScipyOptimizer()                         # Use a SciPy for the optimization
    driver.options['optimizer'] = 'SLSQP'                           # Use the SQP algorithm
    driver.options['disp'] = True                                   # Print the result
    driver.opt_settings = {'disp': True, 'iprint': 2}               # Display iterations

    # 4. Setup the Problem
    prob.setup()                                                    # Call the OpenMDAO setup() method
    model.coupled_group.linear_solver.options['maxiter'] = 17       # Increase maxiter of the linear solver
    model.coupled_group.nonlinear_solver.options['maxiter'] = 17    # Increase maxiter of the nonlinear solver
    prob.run_model()                                                # Run the model once to init. the variables
    model.initialize_from_xml('sellar-input.xml')                   # Set the initial values from an XML file

    # 5. Create and attach some Recorders (Optional)
    """
    from openlego.recorders import NormalizedDesignVarPlotter, ConstraintsPlotter, SimpleObjectivePlotter
    
    desvar_plotter = NormalizedDesignVarPlotter()                   # Create a plotter for the design variables
    desvar_plotter.options['save_on_close'] = True                  # Should this plot be saved automatically?
    desvar_plotter.save_settings['path'] = 'desvar.png'             # Set the filename of the image file

    convar_plotter = ConstraintsPlotter()                           # Create a plotter for the constraints
    convar_plotter.options['save_on_close'] = True                  # Should this plot be saved automatically?
    convar_plotter.save_settings['path'] = 'convar.png'             # Set the filename of the image file

    objvar_plotter = SimpleObjectivePlotter()                       # Create a plotter for the objective
    objvar_plotter.options['save_on_close'] = True                  # Should this plot be saved automatically?
    objvar_plotter.save_settings['path'] = 'objvar.png'             # Set the filename of the image file

    driver.add_recorder(desvar_plotter)                             # Attach the design variable plotter
    driver.add_recorder(convar_plotter)                             # Attach the constraint variable plotter
    driver.add_recorder(objvar_plotter)                             # Attach the objective variable plotter
    """

    # 6. Solve the Problem
    prob.run_driver()                                               # Run the optimization

    # 7. Print results
    from .kb import x_f1, x_x1, x_z1, x_z2, x_y1, x_y2, x_g1, x_g2
    print('Optimum found! Objective function value: f = {}'.format(prob[x_f1]))
    print('Design variables at optimum: x = {}, z1 = {}, z2 = {}'.format(prob[x_x1], prob[x_z1], prob[x_z2]))
    print('Coupling variables at optimum: y1 = {}, y2 = {}'.format(prob[x_y1], prob[x_y2]))
    print('Constraints at optimum: g1 = {}, g2 = {}'.format(prob[x_g1], prob[x_g2]))

    # 8. Cleanup the Problem afterwards
    prob.cleanup()                                                  # Clear all resources and close the plots
    model.invalidate()                                              # Clear the cached properties of the LEGOModel


class TestSellar(unittest.TestCase):

    def test_mdf_gs(self):
        """Solve the Sellar problem using the MDF architecture and a Gauss-Siedel converger."""
        solve_sellar('sellar-MDG_MDF-GS.xml')

    def test_mdf_j(self):
        """Solve the Sellar problem using the MDF architecture and a Jacobi converger."""
        solve_sellar('sellar-MDG_MDF-J.xml')

    def test_idf(self):
        """Solve the Sellar problem using the IDF architecture."""
        solve_sellar('sellar-MDG_IDF.xml')


if __name__ == '__main__':
    unittest.main()
