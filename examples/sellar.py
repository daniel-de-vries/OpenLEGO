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

This file contains the code to create and run the test Sellar case.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

from typing import Optional

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))


def kb_deploy():
    # type: () -> str
    """ Deploy the knowledge base for the sellar problem and returns the path of the base.

    :returns: path to the deployed knowledge base
    """
    from examples.kb.kb_sellar import deploy
    deploy()
    return os.path.join(dir_path, 'kb', 'kb_sellar')


def kb_to_cmdows(kb_path, out_path, create_pdfs=False, open_pdfs=False, create_vistoms=False):
    # type: (str, str, Optional[bool], Optional[bool], Optional[bool]) -> str
    """ Uses KADMOS to transform the sellar knowledge base into a CMDOWS file.

    :param kb_path: path to the knowledge base
    :param out_path: path to the output directory
    :param create_pdfs: set to True to create PDFs with XDSMs
    :param open_pdfs: set to True to open PDFs as they are created
    :param create_vistoms: set to True to create VISTOMS visualization package
    :returns: path to the output CMDOWS file
    """
    from kadmos.graph import FundamentalProblemGraph
    from kadmos.knowledgebase import KnowledgeBase
    from kadmos.utilities.general import get_mdao_setup
    from examples.kb.kb_sellar import x_z1, x_z2, x_x1, x_f1, x_g1, x_g2

    # KB
    kb_path = os.path.split(kb_path)
    kb = KnowledgeBase(kb_path[0], kb_path[1])

    # RCG
    rcg = kb.get_rcg(name='rcg')
    rcg_order = ['D1', 'D2', 'F1', 'G1', 'G2']

    # FPG
    fpg = FundamentalProblemGraph(rcg)
    fpg_order = rcg_order[:]

    mdao_definition = 'MDF-GS'
    mdao_architecture, convergence_type, allow_unconverged_couplings = get_mdao_setup(mdao_definition)
    pf = 'problem_formulation'
    fpg.graph[pf] = dict()
    fpg.graph[pf]['function_order'] = fpg_order
    fpg.graph[pf]['mdao_architecture'] = mdao_architecture
    fpg.graph[pf]['convergence_type'] = convergence_type
    fpg.graph[pf]['allow_unconverged_couplings'] = allow_unconverged_couplings
    fpg.make_all_variables_valid()

    fpg.mark_as_design_variable([x_z1, x_z2, x_x1],
                                lower_bounds=[-10., 0., 0.],
                                nominal_values=[1., 5., 5.],
                                upper_bounds=[10., 10., 10.])

    special_output_nodes = [x_f1, x_g1, x_g2]

    objective = special_output_nodes[0]
    fpg.mark_as_objective(objective)

    constraints = special_output_nodes[1:]
    con_lower_bounds = len(constraints) * [1e99]
    con_upper_bounds = len(constraints) * [0.]
    fpg.mark_as_constraint(constraints, lower_bounds=con_lower_bounds, upper_bounds=con_upper_bounds)

    output_nodes = fpg.find_all_nodes(subcategory='all outputs')
    for output_node in output_nodes:
        if output_node not in special_output_nodes:
            fpg.remove_node(output_node)

    fpg.add_function_problem_roles()

    # MDG and MPG
    mdg = fpg.get_mdg(name='MDG1')
    mpg = fpg.get_mpg(name='MPG1', mdg=mdg)

    mdg_order = fpg_order[:]
    mdg_order.insert(0, 'Optimizer')
    mdg_order.insert(1, 'Converger')

    cmdows_file = 'MDG_' + mdao_definition
    mdg.save(cmdows_file,
             file_type='cmdows',
             destination_folder=out_path,
             mpg=mpg,
             description='Sellar problem MPG file',
             creator='D. de Vries',
             version='0.1',
             pretty_print=True,
             convention=False)

    # Visualizations
    if create_pdfs:
        rcg.create_dsm(file_name='rcg',
                       include_system_vars=True,
                       summarize_vars=False,
                       function_order=rcg_order,
                       open_pdf=open_pdfs,
                       destination_folder=out_path)

        fpg.create_dsm('fpg_' + mdao_definition,
                       include_system_vars=True,
                       summarize_vars=False,
                       open_pdf=open_pdfs,
                       destination_folder=out_path)

        mdg.create_dsm('XDSM_' + mdao_definition,
                       mpg=mpg,
                       summarize_vars=False,
                       open_pdf=open_pdfs,
                       destination_folder=out_path)

    if create_vistoms:
        fpg.graph['description'] = 'FPG_' + str(mdao_architecture) + '_' + str(convergence_type)
        mdg.graph['description'] = 'XDSM_' + mdao_definition

        vistoms_dir = os.path.join(out_path, 'vistoms')
        rcg.vistoms_create(vistoms_dir, function_order=rcg_order)
        fpg.vistoms_add(vistoms_dir, function_order=fpg_order)
        mdg.vistoms_add(vistoms_dir, mpg, function_order=mdg_order)

    return os.path.join(out_path, cmdows_file + '.xml')


def cmdows_to_openmdao(cmdows_path, kb_path, data_folder=None, base_xml_file=None):
    # type: (str, str, Optional[str], Optional[str]) -> CMDOWSProblem
    """ Transforms a CMDOWS file into an OpenMDAO Problem.

    :param cmdows_path: path to the CMDOWS file
    :param kb_path: path to the knowledge base
    :param data_folder: path to the data folder in which to store all files and output from the problem
    :param base_xml_file: path to a base XML file to update with the problem data
    :returns: instance of openlego.CMDOWS.CMDOWSProblem representing the CMDOWS file
    """
    from openlego.CMDOWSProblem import CMDOWSProblem
    from kadmos.cmdows import CMDOWS
    problem = CMDOWSProblem(data_folder=data_folder, base_xml_file=base_xml_file)
    problem.cmdows = CMDOWS(cmdows_path)
    problem.kb_path = kb_path
    return problem


def generate_init_xml(xml_path, z1, z2, x1):
    # type: (str, float, float, float) -> None
    """ Generates the initialization XML file for the reference Sellar problem.

    :param xml_path: path to the XML file
    :param z1: initial value of the z1 parameter
    :param z2: initial value of the z2 parameter
    :param x1: initial value of the x1 parameter
    """
    from openlego.xmlutils import xml_safe_create_element
    from examples.kb.kb_sellar import root_tag, x_x1, x_z1, x_z2
    from lxml import etree

    root = etree.Element(root_tag)
    doc = etree.ElementTree(root)

    xml_safe_create_element(doc, x_x1, x1)
    xml_safe_create_element(doc, x_z1, z1)
    xml_safe_create_element(doc, x_z2, z2)

    doc.write(xml_path, encoding='utf-8', pretty_print=True, xml_declaration=True)


if __name__ == '__main__':
    from shutil import copyfile
    from openmdao.api import view_model
    from openlego.Recorders import NormalizedDesignVarPlotter, ConstraintsPlotter, SimpleObjectivePlotter

    out = os.path.join(dir_path, 'output')
    xml = os.path.join(out, 'input.xml')
    base_file = os.path.join(out, 'base.xml')

    kb_path = kb_deploy()
    cmdows_path = kb_to_cmdows(kb_path, out, False, False, False)
    cmdows_problem = cmdows_to_openmdao(cmdows_path, kb_path, out, base_file)

    desvar_plotter = NormalizedDesignVarPlotter(cmdows_problem.driver)
    desvar_plotter.options['save_on_close'] = True
    desvar_plotter.save_settings['path'] = os.path.join(out, 'desvar.png')

    constr_plotter = ConstraintsPlotter(cmdows_problem.driver)
    constr_plotter.options['save_on_close'] = True
    constr_plotter.save_settings['path'] = os.path.join(out, 'constr.png')

    objvar_plotter = SimpleObjectivePlotter(cmdows_problem.driver)
    objvar_plotter.options['save_on_close'] = True
    objvar_plotter.save_settings['path'] = os.path.join(out, 'objvar.png')

    cmdows_problem.driver.add_recorder(desvar_plotter)
    cmdows_problem.driver.add_recorder(constr_plotter)
    cmdows_problem.driver.add_recorder(objvar_plotter)

    generate_init_xml(xml, 1., 5., 5.)
    copyfile(xml, base_file)

    cmdows_problem.setup()
    view_model(cmdows_problem)

    cmdows_problem.initialize_from_xml(xml)
    cmdows_problem.run()

    cmdows_problem.cleanup()