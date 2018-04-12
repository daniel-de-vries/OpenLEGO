from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import unittest

from kadmos.graph import RepositoryConnectivityGraph, FundamentalProblemGraph
from kadmos.utilities.general import get_mdao_setup

from openmdao.api import Problem, ScipyOptimizer, view_model

from openlego.api import LEGOModel

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


def generate_cmdows(mdao_setup, save_rcg=False, save_fpg=False, save_pdfs=False, xml_dir='.', pdf_dir='.'):
    # XPaths
    x1 = '/data_schema/input/x1'
    x2 = '/data_schema/input/x2'
    y1 = '/data_schema/coupling/y1'
    y2 = '/data_schema/coupling/y2'
    f1 = '/data_schema/output/f1'

    # RCG
    rcg = RepositoryConnectivityGraph(name='Test problem')
    rcg.graph['description'] = 'A simple test problem'
    rcg.add_node('D1', category='function', function_type='regular')
    rcg.add_node('D2', category='function', function_type='regular')
    rcg.add_node('F1', category='function', function_type='regular')

    rcg.add_node(x1, category='variable', label='x1')
    rcg.add_node(x2, category='variable', label='x2')
    rcg.add_node(y1, category='variable', label='y1')
    rcg.add_node(y2, category='variable', label='y2')
    rcg.add_node(f1, category='variable', label='f1')

    rcg.add_edge('D1', y1)
    rcg.add_edge('D2', y2)
    rcg.add_edge('F1', f1)
    rcg.add_edge(x1, 'D1')
    rcg.add_edge(x2, 'D2')
    rcg.add_edge(y1, 'D2')
    rcg.add_edge(y1, 'F1')
    rcg.add_edge(y2, 'D1')
    rcg.add_edge(y2, 'F1')

    rcg.add_equation_labels(rcg.get_function_nodes())
    rcg.add_equation('D1', '2*x1 - y2', 'Python')
    rcg.add_equation('D2', '-2*x2 + y1', 'Python')
    rcg.add_equation('F1', 'y1**2 + y2**2', 'Python')

    function_order = ['D1', 'D2', 'F1']

    if save_pdfs:
        rcg.create_dsm(file_name='RCG',
                       function_order=function_order,
                       include_system_vars=True,
                       destination_folder=pdf_dir,
                       compile_pdf=True)
    if save_rcg:
        rcg.save('RCG',
                 file_type='cmdows',
                 destination_folder=xml_dir,
                 description='RCG CMDOWS file',
                 creator='Daniel',
                 version='0.1',
                 pretty_print=True,
                 integrity=True)

    # FPG
    mdao_architecture, convergence_type, allow_unconverged_couplings = get_mdao_setup(mdao_setup)

    fpg = rcg.deepcopy_as(FundamentalProblemGraph)
    function_order = fpg.get_possible_function_order('single-swap')
    fpg.graph['name'] = 'FPG'
    fpg.graph['description'] = 'Fundamental problem graph of a test case'
    fpg.graph['problem_formulation'] = dict()
    fpg.graph['problem_formulation']['function_order'] = function_order
    fpg.graph['problem_formulation']['mdao_architecture'] = mdao_architecture
    fpg.graph['problem_formulation']['convergence_type'] = convergence_type
    fpg.graph['problem_formulation']['allow_unconverged_couplings'] = allow_unconverged_couplings
    fpg.graph['problem_formulation']['coupled_functions_groups'] = [['D1'], ['D2']]

    fpg.mark_as_objective(f1)
    fpg.mark_as_design_variables([x1, x2], lower_bounds=-10, upper_bounds=10, nominal_values=1.)

    fpg.add_function_problem_roles()

    if save_pdfs:
        fpg.create_dsm(file_name='FPG_{}'.format(mdao_setup),
                       function_order=function_order,
                       include_system_vars=True,
                       destination_folder=pdf_dir)
    if save_fpg:
        fpg.save('FPG_{}'.format(mdao_setup),
                 file_type='cmdows',
                 destination_folder=xml_dir,
                 description='FPG',
                 creator='Daniel',
                 version='0.1',
                 pretty_print=True,
                 integrity=True)

    # MDG/MPG
    mdg = fpg.get_mdg(name='MDG')
    mpg = mdg.get_mpg(name='MPG')
    mdg.graph['name'] = 'XDSM'
    mdg.graph['description'] = 'Solution strategy'

    if save_pdfs:
        mdg.create_dsm(file_name='MDAO_{}'.format(mdao_setup),
                       include_system_vars=True,
                       destination_folder=pdf_dir,
                       mpg=mpg)

    mdao_file = 'MDAO_{}'.format(mdao_setup)
    mdg.save(mdao_file,
             file_type='cmdows',
             destination_folder=xml_dir,
             mpg=mpg,
             description='MDAO_{}'.format(mdao_setup),
             creator='Daniel',
             version='0.1',
             pretty_print=True,
             convention=True)
    return '{}/{}.xml'.format(xml_dir, mdao_file)


def solve_problem(architecture):
    cmdows_file = generate_cmdows(architecture)

    prob = Problem()
    prob.set_solver_print(0)
    
    model = prob.model = LEGOModel(cmdows_file, 'kb', '', 'mathematical_functions_output.xml')
    driver = prob.driver = ScipyOptimizer()

    driver.options['optimizer'] = 'SLSQP'
    driver.options['disp'] = True
    driver.opt_settings = {'disp': True, 'iprint': 2}

    prob.setup()
    view_model(prob)
    prob.cleanup()
    model.invalidate()

    del model


class TestMathFunctions(unittest.TestCase):

        def test_mdf_gs(self):
            """Solve a problem with mathematical functions using the MDF architecture and a Gauss-Siedel converger."""
            solve_problem('MDF-GS')

        def test_idf(self):
            """Solve a problem with mathematical functions using the IDF architecture."""
            solve_problem('IDF')


if __name__ == '__main__':
    unittest.main()
