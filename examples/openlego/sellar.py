from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from openmdao.api import Problem, ScipyOptimizer
from openlego.model import LEGOModel

if __name__ == '__main__':
    # 1. Create a Driver object
    driver = ScipyOptimizer()
    driver.options['optimizer'] = 'SLSQP'
    driver.options['disp'] = True
    driver.options['tol'] = 1.0e-3
    driver.opt_settings = {'disp': True, 'iprint': 2, 'ftol': 1.0e-3}

    # 2. Create the LEGOModel
    model = LEGOModel('sellar_MDG_MDF_GS.xml', '../kb/kb_sellar', '', 'out.xml')

    # 3. Create and setup the problem
    prob = Problem(model=model)
    prob.driver = driver
    prob.set_solver_print(0)
    prob.setup()

    # 4. Run the problem
    prob.run_driver()

    # 5. Cleanup the problem
    prob.cleanup()
