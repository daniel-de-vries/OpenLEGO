from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from openmdao.api import Problem, ScipyOptimizer
from openlego.model import LEGOModel
from openlego.recorders import NormalizedDesignVarPlotter, ConstraintsPlotter, SimpleObjectivePlotter


if __name__ == '__main__':
    # 1. Create a Problem
    prob = Problem()
    prob.set_solver_print(0)

    # 2. Create the LEGOModel
    model = prob.model = LEGOModel('wing_opt_MDG_MDF_GS.xml',           # CMDOWS file
                                   '../../knowledge_bases/wing_opt',    # Knowledge base
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
