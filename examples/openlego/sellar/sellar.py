from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from openmdao.api import ScipyOptimizer
from openlego.problem import CMDOWSProblem
from openlego.util import normalized_to_bounds

if __name__ == '__main__':
    driver = normalized_to_bounds(ScipyOptimizer)()
    driver.options['optimizer'] = 'SLSQP'
    driver.options['maxiter'] = 1000
    driver.options['disp'] = True
    driver.options['tol'] = 1.0e-3
    driver.opt_settings = {'disp': True, 'iprint': 2, 'ftol': 1.0e-3}

    prob = CMDOWSProblem('sellar_MDG_MDF_GS.xml', '../../knowledge_bases/sellar', driver, '', 'sellar_output.xml')
    prob.setup()
    prob.run()
    prob.cleanup()
