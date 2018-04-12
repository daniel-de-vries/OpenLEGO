from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from openmdao.api import Problem, view_model, ScipyOptimizer

from openlego.api import LEGOModel


if __name__ == '__main__':
    prob = Problem()
    prob.model = model = LEGOModel('Mdao_MDF-GS.xml')
    prob.driver = driver = ScipyOptimizer()

    driver.options['optimizer'] = 'SLSQP'
    # prob.driver.options['maxiter'] = 100
    driver.options['tol'] = 1e-8

    prob.setup()
    prob.set_solver_print(level=0)
    view_model(prob)

    prob.run_driver()
    print('done')
