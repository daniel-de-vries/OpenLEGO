# Imports
from __future__ import absolute_import, division, print_function

import logging
import os
import unittest

from openmdao.api import Problem, ScipyOptimizeDriver

from openlego.core.model import LEGOModel

# Settings for logging
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

# List of MDAO definitions that can be wrapped around the problem
mdao_definitions = ['unconverged-MDA-J',     # 0
                    'unconverged-MDA-GS',   # 1
                    'converged-DOE-GS',     # 2
                    'converged-DOE-J',      # 3
                    'converged-MDA-J',      # 4
                    'converged-MDA-GS',     # 5
                    'MDF-GS',               # 6
                    'MDF-J',                # 7
                    'IDF',                  # 8
                    'CO']                   # 9

def get_loop_items(analyze_mdao_definitions):
    if isinstance(analyze_mdao_definitions, int):
        mdao_defs_loop = [mdao_definitions[analyze_mdao_definitions]]
    elif isinstance(analyze_mdao_definitions, list):
        mdao_defs_loop = [mdao_definitions[i] for i in analyze_mdao_definitions]
    elif isinstance(analyze_mdao_definitions, str):
        if analyze_mdao_definitions == 'all':
            mdao_defs_loop = mdao_definitions
        else:
            raise ValueError('String value {} is not allowed for analyze_mdao_definitions.'.format(analyze_mdao_definitions))
    else:
        raise IOError('Invalid input {} provided of type {}.'.format(analyze_mdao_definitions, type(analyze_mdao_definitions)))
    return mdao_defs_loop


def run_openlego(analyze_mdao_definitions):
    # Check and analyze inputs
    mdao_defs_loop = get_loop_items(analyze_mdao_definitions)

    for mdao_def in mdao_defs_loop:

        print('\n-----------------------------------------------')
        print('Running the OpenLEGO of Mdao_{}.xml...'.format(mdao_def))
        print('------------------------------------------------')
        """Solve the Sellar problem using the given CMDOWS file."""
        # 1. Create Problem
        prob = Problem()  # Create an instance of the Problem class
        prob.set_solver_print(0)  # Turn off printing of solver information

        # 2. Create the LEGOModel
        model = prob.model = LEGOModel(cmdows_path=os.path.join('cmdows_files', 'Mdao_{}.xml'.format(mdao_def)),  # CMDOWS file
                                       data_folder='',  # Output directory
                                       base_xml_file='sellar-output.xml')  # Output file

        # 3. Create the Driver
        driver = prob.driver = ScipyOptimizeDriver()  # Use a SciPy for the optimization
        driver.options['optimizer'] = 'SLSQP'  # Use the SQP algorithm
        driver.options['disp'] = True  # Print the result
        driver.opt_settings = {'disp': True, 'iprint': 2}  # Display iterations

        # 4. Setup the Problem
        prob.setup()  # Call the OpenMDAO setup() method
        prob.run_model()  # Run the model once to init. the variables
        model.initialize_from_xml('sellar-input.xml')  # Set the initial values from an XML file

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
        #prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'ln_cons', 'objs']
        prob.run_driver()  # Run the optimization

        # 7. Print results
        x_f1 = '/dataSchema/analyses/f'
        x_x1 = '/dataSchema/geometry/x1'
        x_z1 = '/dataSchema/geometry/z1'
        x_z2 = '/dataSchema/geometry/z2'
        x_y1 = '/dataSchema/analyses/y1'
        x_y2 = '/dataSchema/analyses/y2'
        x_g1 = '/dataSchema/analyses/g1'
        x_g2 = '/dataSchema/analyses/g2'

        print('Optimum found! Objective function value: f = {}'.format(prob[x_f1]))
        print('Design variables at optimum: x = {}, z1 = {}, z2 = {}'.format(prob[x_x1], prob[x_z1], prob[x_z2]))
        print('Coupling variables at optimum: y1 = {}, y2 = {}'.format(prob[x_y1], prob[x_y2]))
        print('Constraints at optimum: g1 = {}, g2 = {}'.format(prob[x_g1], prob[x_g2]))

        # 8. Cleanup the Problem afterwards
        prob.cleanup()  # Clear all resources and close the plots
        model.invalidate()  # Clear the cached properties of the LEGOModel


class TestSellarMath(unittest.TestCase):

    def test_mdf_gs(self):
        """Solve the Sellar problem using the MDF architecture and a Gauss-Seidel convergence scheme."""
        run_openlego(6)

    def test_mdf_j(self):
        """Solve the Sellar problem using the MDF architecture and a Jacobi converger."""
        run_openlego(7)

    def test_idf(self):
        """Solve the Sellar problem using the IDF architecture."""
        run_openlego(8)


if __name__ == '__main__':
    unittest.main()
