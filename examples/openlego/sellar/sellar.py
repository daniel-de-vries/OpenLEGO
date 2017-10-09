from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from openmdao.api import Problem, ScipyOptimizer
from openlego.model import LEGOModel
from openlego.recorders import NormalizedDesignVarPlotter, ConstraintsPlotter, SimpleObjectivePlotter

if __name__ == '__main__':
    # 0. Select a CMDOWS file
    i = 0
    if i == 0:
        CMDOWS_file = 'sellar-MDG_MDF-GS.xml'                           # MDF with a Gauss-Seidel converger
    elif i == 1:
        CMDOWS_file = 'sellar-MDG_MDF-J.xml'                            # MDF with a Jacobi converger
    else:
        CMDOWS_file = 'sellar-MDG_IDF.xml'                              # IDF architecture

    # 1. Create Problem
    prob = Problem()                                                    # Create an instance of the Problem class
    prob.set_solver_print(0)                                            # Turn off printing of solver information

    # 2. Create the LEGOModel
    model = prob.model = LEGOModel(CMDOWS_file,                         # CMDOWS file
                                   '../../knowledge_bases/sellar',      # Knowledge base path
                                   '',                                  # Output directory
                                   'sellar-output.xml')                 # Output file

    # 3. Create the Driver
    driver = prob.driver = ScipyOptimizer()                             # Use a SciPy for the optimization
    driver.options['optimizer'] = 'SLSQP'                               # Use the SQP algorithm
    driver.options['disp'] = True                                       # Print the result
    driver.opt_settings = {'disp': True, 'iprint': 2}                   # Display iterations

    # 4. Setup the Problem
    prob.setup()                                                        # Call the OpenMDAO setup() method
    model.coupled_group.linear_solver.options['maxiter'] = 17           # Increase maxiter of the linear solver
    model.coupled_group.nonlinear_solver.options['maxiter'] = 17        # Increase maxiter of the nonlinear solver
    prob.run_model()                                                    # Run the model once to initialize the variables
    model.initialize_from_xml('sellar-input.xml')                       # Set the initial values from an XML file

    # 5. Create and attach some Recorders (Optional)
    desvar_plotter = NormalizedDesignVarPlotter()                       # Create a plotter for the design variables
    desvar_plotter.options['save_on_close'] = True                      # Should this plot be saved automatically?
    desvar_plotter.save_settings['path'] = 'desvar.png'                 # Set the filename of the image file

    convar_plotter = ConstraintsPlotter()                               # Create a plotter for the constraint variables
    convar_plotter.options['save_on_close'] = True                      # Should this plot be saved automatically?
    convar_plotter.save_settings['path'] = 'convar.png'                 # Set the filename of the image file

    objvar_plotter = SimpleObjectivePlotter()                           # Create a plotter for the objective variable
    objvar_plotter.options['save_on_close'] = True                      # Should this plot be saved automatically?
    objvar_plotter.save_settings['path'] = 'objvar.png'                 # Set the filename of the image file

    driver.add_recorder(desvar_plotter)                                 # Attach the design variable plotter
    driver.add_recorder(convar_plotter)                                 # Attach the constraint variable plotter
    driver.add_recorder(objvar_plotter)                                 # Attach the objective variable plotter

    # 6. Solve the Problem
    prob.run_driver()                                                   # Run the optimization

    # 7. Cleanup the Problem afterwards
    prob.cleanup()                                                      # Clear all resources and close the plots
