from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from openmdao.api import Problem, ScipyOptimizer
from openlego.model import LEGOModel

if __name__ == '__main__':
    # 1. Create Problem
    prob = Problem()
    prob.set_solver_print(0)

    # 2. Create the LEGOModel
    model = prob.model = LEGOModel('sellar_MDG_MDF_GS.xml', '../kb/kb_sellar', '', 'sellar_output.xml')

    # 3. Create the Driver
    driver = prob.driver = ScipyOptimizer()
    driver.options['optimizer'] = 'SLSQP'
    driver.options['disp'] = True
    driver.options['tol'] = 1.0e-3
    driver.opt_settings = {'disp': True, 'iprint': 2, 'ftol': 1.0e-3}

    # 4. Setup the Problem
    prob.setup()
    prob.run_model()
    model.initialize_from_xml('sellar_input.xml')

    # 5. Solve the Problem
    prob.run_driver()
    prob.cleanup()
