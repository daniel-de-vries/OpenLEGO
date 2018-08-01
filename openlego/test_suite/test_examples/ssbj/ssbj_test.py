# Imports
import os
import logging
import unittest

from openmdao.api import Problem, ScipyOptimizeDriver

from openlego.core.model import LEGOModel
import openlego.test_suite.test_examples.ssbj.kb as kb

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
                    'converged-DOE-GS-FF',    # 8
                    'converged-DOE-GS-LH',    # 9
                    'converged-DOE-GS-MC',    # 10
                    'MDF-GS',                 # 11
                    'MDF-J',                  # 12
                    'IDF',                    # 13
                    'CO',                     # 14
                    'BLISS-2000']             # 15


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
        """Solve the Sellar problem using the given CMDOWS file."""

        # 1. Create Problem
        prob = Problem()  # Create an instance of the Problem class
        prob.set_solver_print(0)  # Turn off printing of solver information

        # 2. Create the LEGOModel
        model = prob.model = LEGOModel(os.path.join('cmdows_files', 'Mdao_{}.xml'.format(mdao_def)),
                                       # CMDOWS file
                                       'kb',  # Knowledge base path
                                       '',  # Output directory
                                       'ssbj-output-{}.xml'.format(mdao_def))  # Output file

        # 3. Create the Driver
        driver = prob.driver = ScipyOptimizeDriver()  # Use a SciPy for the optimization
        driver.options['optimizer'] = 'SLSQP'  # Use the SQP algorithm
        driver.options['disp'] = True  # Print the result
        driver.opt_settings = {'disp': True, 'iprint': 2}  # Display iterations
        driver.options['debug_print'] = ['desvars', 'nl_cons', 'ln_cons', 'objs']

        # 4. Setup the Problem
        prob.setup(mode='fwd')  # Call the OpenMDAO setup() method
        prob.run_model()  # Run the model once to init. the variables
        if mdao_def in ['MDF-GS', 'MDF-J', 'IDF', 'CO', 'BLISS-2000']:
            model.initialize_from_xml('SSBJ-base-mdo.xml')  # Set the initial values from an XML file
        else:
            model.initialize_from_xml('SSBJ-base-mda.xml')

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
        prob.run_driver()  # Run the optimization

        # 7. Print results
        x_R = '/dataSchema/aircraft/other/R'
        x_R__scr = '/dataSchema/scaledData/R/scaler'

        print('Optimum found!')
        print('\nObjective function value: {} = {:.3f} ({:.2f}/{:.2f})'.format(prob.model.objective,
                                                                               prob[prob.model.objective][0],
                                                                               prob[x_R][0], prob[x_R__scr][0]))

        print('\nDesign variables at optimum:')

        def format_expected_float_value(value):
            return '{:10.3f}'.format(value) if isinstance(value, float) else '{:10}'.format(value)

        for des_var, metadata in prob.model.design_vars.items():
            print('{:8} = {:10.3f}     => {:10} < x < {:10}, initial: {:10}'.format(des_var.split('/')[-1],
                                                                                    prob[des_var][0],
                                                                                    format_expected_float_value(
                                                                                        metadata['lower']),
                                                                                    format_expected_float_value(
                                                                                        metadata['upper']),
                                                                                    format_expected_float_value(
                                                                                        metadata['initial'])))

        print('\nConstraint values at optimum:')
        for constraint, metadata in prob.model.constraints.items():
            if metadata['equals'] is not None:
                print('{:8} = {:10.3f}     => c == {:10}'.format(constraint.split('/')[-1],
                                                                 prob[constraint][0],
                                                                 metadata['equals'] if metadata[
                                                                                           'equals'] is not None else ''))
            else:
                print('{:8} = {:10.3f}     => {:10} < c < {:10}'.format(constraint.split('/')[-2],
                                                                        prob[constraint][0],
                                                                        format_expected_float_value(metadata['lower']),
                                                                        format_expected_float_value(metadata['upper'])))

        print('\nAll values at optimum:')
        print('Inputs')
        print('------')
        for input in prob.model._inputs:
            print('{:105} = {:10.3f}'.format(input, prob[input][0]))

        print('\nOutputs')
        print('-------')
        for output in prob.model._outputs:
            print('{:105} = {:10.3f}'.format(output, prob[output][0]))

        # 8. Cleanup the Problem afterwards
        prob.cleanup()  # Clear all resources and close the plots
        model.invalidate()  # Clear the cached properties of the LEGOModel


class TestSsbj(unittest.TestCase):

    def __call__(self, *args, **kwargs):
        kb.deploy()
        super(TestSsbj, self).__call__(*args, **kwargs)

    def test_mdf_gs(self):
        """Solve the SSBJ problem using the MDF architecture and a Gauss-Seidel convergence scheme."""
        run_openlego(11)

    def test_mdf_j(self):
        """Solve the SSBJ problem using the MDF architecture and a Jacobi converger."""
        run_openlego(12)

    def test_idf(self):
        """Solve the SSBJ problem using the IDF architecture."""
        run_openlego(13)

    def __del__(self):
        kb.clean()


if __name__ == '__main__':
    unittest.main()
