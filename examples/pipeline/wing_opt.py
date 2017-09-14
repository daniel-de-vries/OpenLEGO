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

This file contains the code to create and run the test wing optimization case.
"""
from __future__ import absolute_import, division, print_function

import logging
import os
from collections import OrderedDict

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from typing import Optional, List

from openlego.Recorders import BaseIterationPlotter

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

dir_path = os.path.dirname(os.path.realpath(__file__))

m_F_init = 108508.
m_wing_init = 49591.
S_ref_init = 493.59604204
C_L_buffet = 0.525

m_fixed = 107814.
m_fuel_res = 15000.
t_skin = 1.5e-3
rho_skin = 2180.
m_payload = 34000.
m_mlw = 213180.
f_m_sys = 0.27
f_m_wings = 0.7

R = float(14306700.)
SFC = float(1.5E-5)
C_D_fus = float(0.006)
C_D_other = float(0.005)

sigma_yield = float(276E6)
S_wet_0 = float(2 * 383.7)
incidence = float(0.1172)

M_cruise = float(0.85)
H_cruise = float(11277.6)

M_25g = float(0.85)
H_25g = float(3048.)

M_01g = float(0.60)
H_01g = float(0.)

c_0 = np.array([13.7131, 7.2595, 2.7341])
tc_0 = np.array([0.1542, 0.1052, 0.0950])
epsilon_0 = np.array([-0.1039, -0.1826])
b_0 = np.array([12.7178, 22.7016])
Lambda_0 = np.array([0.5435, 0.6077])
Gamma_0 = np.array([0.0508, 0.1167])
xsi_fs_0 = np.array([0.1000, 0.1925, 0.3500])
xsi_rs_0 = np.array([0.6000, 0.8023, 0.6000])
t_fs_0 = np.array([0.00450588, 0.00458215])
t_rs_0 = np.array([0.00450611, 0.00456957])
t_ts_0 = np.array([0.02553329, 0.02237119])
t_bs_0 = np.array([0.02553329, 0.02237119])

c_lim = np.array([1., 15.])
tc_lim = np.array([.04, .20])
epsilon_lim = np.array([-0.25, 0.25])
b_lim = np.array([5., 25.])
Lambda_lim = np.array([0.0, 0.7])
Gamma_lim = np.array([-0.12, 0.12])
xsi_fs_lim = np.array([0.05, 0.40])
xsi_rs_lim = np.array([0.60, 0.90])
t_fs_lim = np.array([0.0010, 0.0300])
t_rs_lim = np.array([0.0010, 0.0300])
t_ts_lim = np.array([0.0010, 0.0300])
t_bs_lim = np.array([0.0010, 0.0300])


def get_variables(n_wing_segments=2):
    # type: (int) -> OrderedDict
    """Get a dictionary with all the input variables for the wing problem for a given number of wing segments.

    Parameters
    ----------
        n_wing_segments : int(2)
            Number of wing segments.

    Returns
    -------
        obj:`OrderedDict`
            Dictionary with the system input variables.
    """
    _c_lim = np.tile(c_lim, (n_wing_segments + 1, 1)).T
    _tc_lim = np.tile(tc_lim, (n_wing_segments + 1, 1)).T
    _epsilon_lim = np.tile(epsilon_lim, (n_wing_segments, 1)).T
    _b_lim = np.tile(b_lim, (n_wing_segments, 1)).T
    _Lambda_lim = np.tile(Lambda_lim, (n_wing_segments, 1)).T
    _Gamma_lim = np.tile(Gamma_lim, (n_wing_segments, 1)).T
    _xsi_fs_lim = np.tile(xsi_fs_lim, (n_wing_segments + 1, 1)).T
    _xsi_rs_lim = np.tile(xsi_rs_lim, (n_wing_segments + 1, 1)).T
    _t_fs_lim = np.tile(t_fs_lim, (n_wing_segments, 1)).T
    _t_rs_lim = np.tile(t_rs_lim, (n_wing_segments, 1)).T
    _t_ts_lim = np.tile(t_ts_lim, (n_wing_segments, 1)).T
    _t_bs_lim = np.tile(t_bs_lim, (n_wing_segments, 1)).T

    if n_wing_segments == 1:
        _b_lim[1, :] *= 2.

        _c_0 = c_0[[0, 2]]
        _tc_0 = tc_0[[0, 2]]
        _epsilon_0 = epsilon_0[[1]]
        _b_0 = np.array([sum(b_0)])
        _Lambda_0 = np.array([np.average(Lambda_0)])
        _Gamma_0 = np.array([np.average(Gamma_0)])
        _xsi_fs_0 = xsi_fs_0[[0, 2]]
        _xsi_rs_0 = xsi_rs_0[[0, 2]]
        _t_fs_0 = np.array([sum(t_fs_0) / 1.2])
        _t_rs_0 = np.array([sum(t_rs_0) / 1.2])
        _t_ts_0 = np.array([sum(t_ts_0) / 1.1])
        _t_bs_0 = np.array([sum(t_bs_0) / 1.1])
    else:
        n_ib = int(n_wing_segments / 2)
        n_ob = n_wing_segments - n_ib

        f_ib = np.linspace(0, 1, n_ib + 1)
        f_ob = np.linspace(0, 1, n_ob + 1)[1:]

        xp = np.array([0, 1])

        _c_0 = np.concatenate((np.interp(f_ib, xp, c_0[:2]), np.interp(f_ob, xp, c_0[1:])))
        _tc_0 = np.concatenate((np.interp(f_ib, xp, tc_0[:2]), np.interp(f_ob, xp, tc_0[1:])))
        _epsilon_0 = np.concatenate((np.interp(f_ib[1:], xp, np.array([0, epsilon_0[0]])),
                                     np.interp(f_ob, xp, epsilon_0)))
        _b_0 = np.concatenate((n_ib * [b_0[0] / float(n_ib)], n_ob * [b_0[1] / float(n_ob)]))
        _Lambda_0 = np.concatenate((n_ib * [Lambda_0[0]], n_ob * [Lambda_0[1]]))
        _Gamma_0 = np.concatenate((n_ib * [Gamma_0[0]], n_ob * [Gamma_0[1]]))
        _xsi_fs_0 = np.concatenate((np.interp(f_ib, xp, xsi_fs_0[:2]), np.interp(f_ob, xp, xsi_fs_0[1:])))
        _xsi_rs_0 = np.concatenate((np.interp(f_ib, xp, xsi_rs_0[:2]), np.interp(f_ob, xp, xsi_rs_0[1:])))
        _t_fs_0 = np.concatenate((n_ib * [t_fs_0[0]], n_ob * [t_fs_0[1]]))
        _t_rs_0 = np.concatenate((n_ib * [t_rs_0[0]], n_ob * [t_rs_0[1]]))
        _t_ts_0 = np.concatenate((n_ib * [t_ts_0[0]], n_ob * [t_ts_0[1]]))
        _t_bs_0 = np.concatenate((n_ib * [t_bs_0[0]], n_ob * [t_bs_0[1]]))

    return OrderedDict([
        ('c',            {'value': _c_0,          'lower': _c_lim[0],         'upper': _c_lim[1]}),
        ('tc',           {'value': _tc_0,         'lower': _tc_lim[0],        'upper': _tc_lim[1]}),
        ('epsilon',      {'value': _epsilon_0,    'lower': _epsilon_lim[0],   'upper': _epsilon_lim[1]}),
        ('b',            {'value': _b_0,          'lower': _b_lim[0],         'upper': _b_lim[1]}),
        ('Lambda',       {'value': _Lambda_0,     'lower': _Lambda_lim[0],    'upper': _Lambda_lim[1]}),
        ('Gamma',        {'value': _Gamma_0,      'lower': _Gamma_lim[0],     'upper': _Gamma_lim[1]}),
        ('xsi_fs',       {'value': _xsi_fs_0,     'lower': _xsi_fs_lim[0],    'upper': _xsi_fs_lim[1]}),
        ('xsi_rs',       {'value': _xsi_rs_0,     'lower': _xsi_rs_lim[0],    'upper': _xsi_rs_lim[1]}),
        ('t_fs',         {'value': _t_fs_0,       'lower': _t_fs_lim[0],      'upper': _t_fs_lim[1]}),
        ('t_rs',         {'value': _t_rs_0,       'lower': _t_rs_lim[0],      'upper': _t_rs_lim[1]}),
        ('t_ts',         {'value': _t_ts_0,       'lower': _t_ts_lim[0],      'upper': _t_ts_lim[1]}),
        ('t_bs',         {'value': _t_bs_0,       'lower': _t_bs_lim[0],      'upper': _t_bs_lim[1]}),
        ('incidence',    {'value': incidence}),
        ('t_skin',       {'value': t_skin}),
        ('rho_skin',     {'value': rho_skin}),
        ('C_D_fus',      {'value': C_D_fus}),
        ('C_D_other',    {'value': C_D_other}),
        ('SFC',          {'value': SFC}),
        ('m_fuel_res',   {'value': m_fuel_res}),
        ('m_fixed',      {'value': m_fixed}),
        ('m_payload',    {'value': m_payload}),
        ('m_mlw',        {'value': m_mlw}),
        ('f_m_sys',      {'value': f_m_sys}),
        ('f_m_wings',    {'value': f_m_wings}),
        ('R',            {'value': R}),
        ('M_cruise',     {'value': M_cruise}),
        ('H_cruise',     {'value': H_cruise}),
        ('n_cruise',     {'value': 1.0}),
        ('M_25g',        {'value': M_25g}),
        ('H_25g',        {'value': H_25g}),
        ('n_25g',        {'value': 2.5}),
        ('M_01g',        {'value': M_01g}),
        ('H_01g',        {'value': H_01g}),
        ('n_01g',        {'value': -1.}),
        ('m_F_init',     {'value': m_F_init}),
        ('m_wing_init',  {'value': m_wing_init}),
        ('S_ref_init',   {'value': S_ref_init}),
        ('sigma_yield',  {'value': sigma_yield}),
        ('C_L_buffet',   {'value': C_L_buffet})
    ])


def kb_deploy(n_wing_segments=2, n_load_cases=3):
    # type: (int, int) -> str
    """Deploy the knowledge base for the wing optimization problem and return the path of the base.

    Parameters
    ----------
        n_wing_segments : int(2)
            Number of wing segments.

        n_load_cases : int(3)
            Number of load cases.

    Returns
    -------
        str
            Path to the folder of the deployed knowledge base.
    """
    from examples.kb.kb_wing_opt import deploy
    deploy(n_wing_segments, n_load_cases)
    return os.path.join(dir_path, 'kb', 'kb_wing_opt')


def kb_to_cmdows(kb_path, out_path, n_wing_segments=2, create_pdfs=False, open_pdfs=False, create_vistoms=False):
    # type: (str, str, int, bool, bool, bool) -> str
    """Use KADMOS to transform the wing design knowledge base into a CMDOWS file.

    Parameters
    ----------
        kb_path : str
            Path to the folder of the knowledge base.

        out_path : str
            Path to the output folder.

        n_wing_segments : int(2)
            Number of wing segments.

        create_pdfs : bool(False)
            Set to `True` to create PDF files of the XDSM diagrams created by KADMOS.

        open_pdfs : bool(False)
            Set to `True` to open the PDF files as they are created.

        create_vistoms : bool(False)
            Set to `True` to create an interactive visualization package of the problem.

    Returns
    -------
        str
            Path to the CMDOWS file created by KADMOS.
    """
    from kadmos.graph import FundamentalProblemGraph
    from kadmos.knowledgebase import KnowledgeBase
    from kadmos.utilities.general import get_mdao_setup
    from examples.kb.kb_wing_opt.disciplines.xpaths import x_c, x_epsilon, x_b, \
        x_xsi_fs, x_xsi_rs, x_t_fs, x_t_rs, x_t_ts, x_t_bs, \
        x_obj_m_fuel, x_con_sigmas, x_con_exposed_area

    # Get dict of variables
    variables = get_variables(n_wing_segments)

    # KB
    kb_path = os.path.split(kb_path)
    kb = KnowledgeBase(kb_path[0], kb_path[1])

    # RCG
    rcg = kb.get_rcg(name='rcg')
    rcg_order = ['WOM', 'dSMI', 'dSAMI', 'dSAA', 'dSSA', 'dSLD', 'dLC', 'FWE',
                 'ConstraintFunctions', 'ObjectiveFunctions']

    # FPG
    fpg = FundamentalProblemGraph(rcg)
    fpg.remove_function_nodes('dSLD')
    fpg_order = rcg_order[:]
    fpg_order.remove('dSLD')

    mdao_definition = 'MDF-GS'
    mdao_architecture, convergence_type, allow_unconverged_couplings = get_mdao_setup(mdao_definition)
    pf = 'problem_formulation'
    fpg.graph[pf] = dict()
    fpg.graph[pf]['function_order'] = fpg_order
    fpg.graph[pf]['mdao_architecture'] = mdao_architecture
    fpg.graph[pf]['convergence_type'] = convergence_type
    fpg.graph[pf]['allow_unconverged_couplings'] = allow_unconverged_couplings
    fpg.make_all_variables_valid()

    desvars = [x_c, x_epsilon, x_b, x_xsi_fs, x_xsi_rs, x_t_fs, x_t_rs, x_t_ts, x_t_bs]
    # desvars = [x_t_fs, x_t_rs, x_t_ts, x_t_bs]
    lower_bounds = len(desvars) * [None]
    upper_bounds = len(desvars) * [None]
    nominal_values = len(desvars) * [None]
    for i in range(len(desvars)):
        desvar = desvars[i].split('/')[-1]
        lower_bounds[i] = variables[desvar]['lower'].tolist()
        upper_bounds[i] = variables[desvar]['upper'].tolist()
        nominal_values[i] = variables[desvar]['value'].tolist()

    fpg.mark_as_design_variable(desvars,
                                lower_bounds=lower_bounds,
                                nominal_values=nominal_values,
                                upper_bounds=upper_bounds)

    special_output_nodes = [
        # x_obj_m_wing,
        x_obj_m_fuel,
        x_con_sigmas[0],
        x_con_sigmas[1],
        x_con_sigmas[2],
        x_con_sigmas[3],
        x_con_exposed_area]
        # x_con_buffet]

    objective = special_output_nodes[0]
    fpg.mark_as_objective(objective)

    constraints = special_output_nodes[1:]
    con_lower_bounds = len(constraints) * [-1e99]
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
    mdg_order.insert(4, 'Converger')

    cmdows_file = 'MDG_' + mdao_definition
    mdg.save(cmdows_file,
             file_type='cmdows',
             destination_folder=out_path,
             mpg=mpg,
             description='Wing optimization MPG file',
             creator='D. de Vries',
             version='0.1',
             pretty_print=True,
             convention=False)

    # Visualizations
    if create_pdfs:
        rcg.create_dsm(file_name='rcg',
                       include_system_vars=True,
                       summarize_vars=True,
                       function_order=rcg_order,
                       open_pdf=open_pdfs,
                       destination_folder=out_path)

        fpg.create_dsm('fpg_' + mdao_definition,
                       include_system_vars=True,
                       summarize_vars=True,
                       open_pdf=open_pdfs,
                       destination_folder=out_path)

        mdg.create_dsm('XDSM_' + mdao_definition,
                       mpg=mpg,
                       summarize_vars=True,
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


def generate_init_xml(xml_path, n_wing_segments=2, load_cases=None):
    # type: (str, int, Optional[List[(float, float, float)]]) -> None
    """Generate the initialization XML file for the reference wing optimization problem.

    Parameters
    ----------
        xml_path : str
            Path of the initialization XML file.

        n_wing_segments : int(2)
            Number of wing segments.

        load_cases : list of (float, float, float), optional
            List of load cases in the form of tuples with (Mach number, altitude, load factor).
    """
    from openlego.xmlutils import xml_safe_create_element
    from examples.kb.kb_wing_opt.disciplines.xpaths import x_m_fixed, x_m_payload, x_m_mlw, \
        x_f_m_sys, x_f_m_wings, x_R, x_SFC, x_m_fuel_res, x_CDfus, x_CDother, \
        x_m_fuel_init, x_sigma_yield, x_m_wing_init, x_S_ref_init, x_CL_buffet, \
        x_M, x_H, x_n, x_ml_timeout, \
        x_c, x_tc, x_epsilon, x_b, x_Lambda, x_Gamma, x_xsi_fs, \
        x_xsi_rs, x_t_fs, x_t_rs, x_t_ts, x_t_bs, x_incidence, x_t_skin, x_rho_skin
    from lxml import etree
    variables = get_variables(n_wing_segments)

    root = etree.Element('cpacs')
    doc = etree.ElementTree(root)

    xml_safe_create_element(doc, x_c, variables['c']['value'])
    xml_safe_create_element(doc, x_tc, variables['tc']['value'])
    xml_safe_create_element(doc, x_epsilon, variables['epsilon']['value'])
    xml_safe_create_element(doc, x_b, variables['b']['value'])
    xml_safe_create_element(doc, x_Lambda, variables['Lambda']['value'])
    xml_safe_create_element(doc, x_Gamma, variables['Gamma']['value'])
    xml_safe_create_element(doc, x_xsi_fs, variables['xsi_fs']['value'])
    xml_safe_create_element(doc, x_xsi_rs, variables['xsi_rs']['value'])
    xml_safe_create_element(doc, x_t_fs, variables['t_fs']['value'])
    xml_safe_create_element(doc, x_t_rs, variables['t_rs']['value'])
    xml_safe_create_element(doc, x_t_ts, variables['t_ts']['value'])
    xml_safe_create_element(doc, x_t_bs, variables['t_bs']['value'])

    xml_safe_create_element(doc, x_incidence, variables['incidence']['value'])
    xml_safe_create_element(doc, x_t_skin, variables['t_skin']['value'])
    xml_safe_create_element(doc, x_rho_skin, variables['rho_skin']['value'])

    xml_safe_create_element(doc, x_m_fixed, variables['m_fixed']['value'])
    xml_safe_create_element(doc, x_m_payload, variables['m_payload']['value'])
    xml_safe_create_element(doc, x_m_mlw, variables['m_mlw']['value'])
    xml_safe_create_element(doc, x_f_m_sys, variables['f_m_sys']['value'])
    xml_safe_create_element(doc, x_f_m_wings, variables['f_m_wings']['value'])

    xml_safe_create_element(doc, x_R, variables['R']['value'])
    xml_safe_create_element(doc, x_SFC, variables['SFC']['value'])
    xml_safe_create_element(doc, x_m_fuel_res, variables['m_fuel_res']['value'])
    xml_safe_create_element(doc, x_CDfus, variables['C_D_fus']['value'])
    xml_safe_create_element(doc, x_CDother, variables['C_D_other']['value'])

    if load_cases is None:
        xml_safe_create_element(doc, x_M % 1, variables['M_cruise']['value'])
        xml_safe_create_element(doc, x_H % 1, variables['H_cruise']['value'])
        xml_safe_create_element(doc, x_n % 1, variables['n_cruise']['value'])
        xml_safe_create_element(doc, x_ml_timeout % 1, 1800.)

        xml_safe_create_element(doc, x_M % 2, variables['M_25g']['value'])
        xml_safe_create_element(doc, x_H % 2, variables['H_25g']['value'])
        xml_safe_create_element(doc, x_n % 2, variables['n_25g']['value'])
        xml_safe_create_element(doc, x_ml_timeout % 2, 1800.)

        xml_safe_create_element(doc, x_M % 3, variables['M_01g']['value'])
        xml_safe_create_element(doc, x_H % 3, variables['H_01g']['value'])
        xml_safe_create_element(doc, x_n % 3, variables['n_01g']['value'])
        xml_safe_create_element(doc, x_ml_timeout % 3, 1800.)
    else:
        for index, load_case in enumerate(load_cases):
            xml_safe_create_element(doc, x_M % (index + 1), load_case[0])
            xml_safe_create_element(doc, x_H % (index + 1), load_case[1])
            xml_safe_create_element(doc, x_n % (index + 1), load_case[2])
            xml_safe_create_element(doc, x_ml_timeout % (index + 1), 1800.)

    xml_safe_create_element(doc, x_m_fuel_init, variables['m_F_init']['value'])
    xml_safe_create_element(doc, x_m_wing_init, variables['m_wing_init']['value'])
    xml_safe_create_element(doc, x_S_ref_init, variables['S_ref_init']['value'])
    xml_safe_create_element(doc, x_sigma_yield, variables['sigma_yield']['value'])
    xml_safe_create_element(doc, x_CL_buffet, variables['C_L_buffet']['value'])

    doc.write(xml_path, encoding='utf-8', pretty_print=True, xml_declaration=True)


class WingDesignPlotter(BaseIterationPlotter):
    """Specialized `BaseIterationPlotter` displaying a top-view of the wing, the spars, and an outline of the original.

    Attributes
    ----------
        p_c, p_epsilon, p_b, p_Lambda, p_Gamma, p_incidence, p_xsi_fs, p_xsi_rs : str
            OpenMDAO parameter names of the chord, twist, span, sweep, dihedral, incidence, front spar location, and
            rear spar location variables.

        ax : Axes
            Axes of the figure.

        first_run : bool
            Flag to indicate whether it is the first run of this `Recorder`.
            Flipped to `False` after first run.

        n_wing_segments : int
            Number of wing segments.

        x_outline_0 : :obj:`np.ndarray`
            Coordinates of the vertices of the outline of the wing.

        x_fs_0, x_rs_0 : :obj:`np.ndarray`
            Coordinates of the front and rear spar vertices.
    """

    from examples.kb.kb_wing_opt.disciplines.xpaths import x_c, x_epsilon, x_b,\
        x_Lambda, x_Gamma, x_incidence, x_xsi_fs, x_xsi_rs
    from openlego.xmlutils import xpath_to_param

    p_c = xpath_to_param(x_c)
    p_epsilon = xpath_to_param(x_epsilon)
    p_b = xpath_to_param(x_b)
    p_Lambda = xpath_to_param(x_Lambda)
    p_Gamma = xpath_to_param(x_Gamma)
    p_incidence = xpath_to_param(x_incidence)
    p_xsi_fs = xpath_to_param(x_xsi_fs)
    p_xsi_rs = xpath_to_param(x_xsi_rs)

    def __init__(self):
        """Initialize the `WingDesignPlotter`."""
        super(WingDesignPlotter, self).__init__()

        self.ax = None
        self.first_run = True

        self.n_wing_segments = None
        self.x_outline_0 = None
        self.x_fs_0 = None
        self.x_rs_0 = None

    def init_fig(self, fig):
        """Add axes and set the aspect ratio of the figure to equal.

        Parameters
        ----------
            fig : `Figure`
                Figure handle of the plot.
        """
        self.ax = fig.add_subplot(111)
        self.ax.set_aspect('equal', 'box')

    def _update_plot(self, params, unknowns, resids, metadata):
        """Update the plot of the wing shape for the next iteration.

        Parameters
        ----------
            params : dict
                Dictionary containing the ``OpenMDAO`` parameters.

            unknowns : dict
                Dictionary containing the ``OpenMDAO`` unknowns.

            resids : dict
                Dictionary containing the ``OpenMDAO`` residuals.

            metadata : dict
                Dictionary containing the ``OpenMDAO`` metadata.
         """
        if self.first_run:
            self.n_wing_segments = unknowns[self.p_b].size

        dx_c4 = np.zeros((3, self.n_wing_segments))
        dx_c4[1, :] = unknowns[self.p_b]

        x_c4 = np.zeros((3, self.n_wing_segments + 1))
        c_sweep, s_sweep = np.cos(unknowns[self.p_Lambda]), np.sin(unknowns[self.p_Lambda])
        c_dihed, s_dihed = np.cos(unknowns[self.p_Gamma]), np.sin(unknowns[self.p_Gamma])

        for i in range(self.n_wing_segments):
            rot_sweep = np.matrix([(c_sweep[i], s_sweep[i], 0), (-s_sweep[i], c_sweep[i], 0), (0, 0, 1)])
            rot_dihed = np.matrix([(1, 0, 0), (0, c_dihed[i], -s_dihed[i]), (0, s_dihed[i], c_dihed[i])])

            x_c4[:, i + 1] = np.matmul(rot_dihed*rot_sweep, dx_c4[:, i]) + x_c4[:, i]

        dx_le = np.zeros((3, self.n_wing_segments + 1))
        dx_te = np.zeros((3, self.n_wing_segments + 1))
        dx_le[0, :] = -.25*unknowns[self.p_c]
        dx_te[0, :] = .75*unknowns[self.p_c]

        x_le = np.zeros((3, self.n_wing_segments + 1))
        x_te = np.zeros((3, self.n_wing_segments + 1))

        x_fs = np.zeros((3, self.n_wing_segments + 1))
        x_rs = np.zeros((3, self.n_wing_segments + 1))

        dx_sp = np.zeros((3, 1))

        twists = np.concatenate(([0.], unknowns[self.p_epsilon])) + unknowns[self.p_incidence]
        c_twist, s_twist = np.cos(twists), np.sin(twists)
        for i in range(self.n_wing_segments + 1):
            rot_twist = np.matrix([(c_twist[i], 0, s_twist[i]), (0, 1, 0), (-s_twist[i], 0, c_twist[i])])

            x_le[:, i] = np.matmul(rot_twist, dx_le[:, i]) + x_c4[:, i]
            x_te[:, i] = np.matmul(rot_twist, dx_te[:, i]) + x_c4[:, i]

            dx_sp[0] = (unknowns[self.p_xsi_fs][i] - 0.25) * unknowns[self.p_c][i]
            x_fs[:, i] = np.matmul(rot_twist, dx_sp[:, 0]) + x_c4[:, i]

            dx_sp[0] = (unknowns[self.p_xsi_rs][i] - 0.25) * unknowns[self.p_c][i]
            x_rs[:, i] = np.matmul(rot_twist, dx_sp[:, 0]) + x_c4[:, i]

        x_outline = np.concatenate((x_le, np.fliplr(x_te)), axis=1)

        x = np.array([x_outline[0, :]]).T
        y = np.array([x_outline[1, :]]).T

        polygon = Polygon(np.concatenate((x, y), axis=1), True)
        p = PatchCollection([polygon], facecolors='red', alpha=0.4)

        self.ax.clear()
        if self.first_run:
            self.x_outline_0 = x_outline.copy()
            self.x_fs_0 = x_fs.copy()
            self.x_rs_0 = x_rs.copy()
        else:
            self.ax.plot(self.x_outline_0[0, :], self.x_outline_0[1, :], color='grey')

        self.ax.add_collection(p)
        self.ax.plot(x_fs[0, :], x_fs[1, :], color='red', label='Front spar')
        self.ax.plot(x_rs[0, :], x_rs[1, :], color='blue', label='Rear spar')
        self.ax.plot(x_outline[0, :], x_outline[1, :], color='black')
        self.ax.set_xlabel('x, [m]')
        self.ax.set_ylabel('y, [m]')

        if not self.first_run:
            x_ = np.append(x, np.array([self.x_outline_0[0, :]]).T)
            y_ = np.append(y, np.array([self.x_outline_0[1, :]]).T)
            self.ax.set_xlim([np.min(x_), np.max(x_)])
            self.ax.set_ylim([np.min(y_), np.max(y_)])
        else:
            self.ax.set_xlim([np.min(x), np.max(x)])
            self.ax.set_ylim([np.min(y), np.max(y)])

        self.ax.set_aspect('equal', adjustable='box')

        self.first_run = False


if __name__ == '__main__':
    from shutil import copyfile
    from openlego.xmlutils import xpath_to_param
    from examples.kb.kb_wing_opt.disciplines.xpaths import x_m_fuel, x_m_wing, \
        x_t_fs, x_t_rs, x_t_ts, x_t_bs, x_con_exposed_area
    from openmdao.api import ScipyOptimizer
    from openmdao.recorders.sqlite_recorder import SqliteRecorder
    from openlego.Recorders import NormalizedDesignVarPlotter, ConstraintsPlotter, SimpleObjectivePlotter, VOIPlotter
    from openlego.CMDOWSProblem import CMDOWSProblem
    from openlego.BoundsNormalizedDriver import normalized_to_bounds

    # Problem settings
    n_ws = 2
    n_lc = 3

    # Output paths and files
    out = os.path.join(dir_path, 'output')
    xml = os.path.join(out, 'input.xml')
    base_file = os.path.join(out, 'base.xml')
    data_file = os.path.join(out, 'data.db')

    # Obtain a dictionary of the variables for the given amount of wing segments
    variables = get_variables(n_wing_segments=n_ws)

    # Generate the input XML file for the problem
    generate_init_xml(xml, n_wing_segments=n_ws)
    copyfile(xml, base_file)

    # Create a driver for the problem
    driver = normalized_to_bounds(ScipyOptimizer)()
    driver.options['optimizer'] = 'SLSQP'
    driver.options['maxiter'] = 1000
    driver.options['disp'] = True
    driver.options['tol'] = 1.0e-3
    driver.opt_settings = {'disp': True, 'iprint': 2, 'ftol': 1.0e-3}

    # Pipeline: Knowledgebase -> KADMOS -> CMDOWS file -> OpenMDAO Problem
    kb_path = kb_deploy(n_ws, n_lc)
    cmdows_path = kb_to_cmdows(kb_path, out, n_ws)
    cmdows_problem = CMDOWSProblem(cmdows_path, kb_path, driver, out, base_file)

    # Manually fix the exposed area equality contraint
    cmdows_problem.driver._cons[xpath_to_param(x_con_exposed_area)]['lower'] = None
    cmdows_problem.driver._cons[xpath_to_param(x_con_exposed_area)]['upper'] = None
    cmdows_problem.driver._cons[xpath_to_param(x_con_exposed_area)]['equals'] = 0.

    # Create and setup all recorders for the OpenMDAO Problem
    recorder = SqliteRecorder(data_file)
    cmdows_problem.driver.add_recorder(recorder)

    desvar_plotter = NormalizedDesignVarPlotter(cmdows_problem.driver)
    desvar_plotter.options['save_on_close'] = True
    desvar_plotter.save_settings['path'] = os.path.join(out, 'desvar.png')

    constr_plotter = ConstraintsPlotter(cmdows_problem.driver)
    constr_plotter.options['save_on_close'] = True
    constr_plotter.save_settings['path'] = os.path.join(out, 'constr.png')

    objvar_plotter = SimpleObjectivePlotter(cmdows_problem.driver)
    objvar_plotter.options['save_on_close'] = True
    objvar_plotter.save_settings['path'] = os.path.join(out, 'objvar.png')

    voi_plotter = VOIPlotter()
    voi_plotter.options['save_on_close'] = True
    voi_plotter.save_settings['path'] = os.path.join(out, 'vois.png')
    voi_plotter.options['includes'] = [xpath_to_param(x_m_fuel), xpath_to_param(x_m_wing)]
    voi_plotter.options['legend'] = ['Fuel mass', 'Wing mass']
    voi_plotter.options['labels'] = ['m_F, [kg]', 'm_wing, [kg]']

    wing_des_plotter = WingDesignPlotter()
    wing_des_plotter.options['save_on_close'] = True
    wing_des_plotter.save_settings['path'] = os.path.join(out, 'wing.png')

    cmdows_problem.driver.add_recorder(desvar_plotter)
    cmdows_problem.driver.add_recorder(constr_plotter)
    cmdows_problem.driver.add_recorder(objvar_plotter)
    cmdows_problem.driver.add_recorder(voi_plotter)
    cmdows_problem.driver.add_recorder(wing_des_plotter)

    # Setup the OpenMDAO Problem
    cmdows_problem.setup()
    cmdows_problem.initialize_from_xml(xml)

    # Run the problem

    # cmdows_problem.run_once()
    # print('Initial value m_F: %f kg' % cmdows_problem.root.unknowns[xpath_to_param(x_m_fuel)])
    # print('Initial value m_wing: %f kg' % cmdows_problem.root.unknowns[xpath_to_param(x_m_wing)])

    # cmdows_problem.run()
    # print('Final value m_F: %f kg' % cmdows_problem.root.unknowns[xpath_to_param(x_m_fuel)])

    # cmdows_problem.run_once()
    # print('Initial value m_wing: %f kg' % cmdows_problem.root.unknowns[xpath_to_param(x_m_wing)])

    cmdows_problem.run()
    print('Optimal value m_fuel: %f kg' % cmdows_problem.root.unknowns[xpath_to_param(x_m_fuel)])
    print('Final value m_wing: %f kg' % cmdows_problem.root.unknowns[xpath_to_param(x_m_wing)])
    print('Optimal thicknesses for minimal m_wing: t_fs = %s, t_rs = %s, t_ts = %s, t_bs = %s' % (
        str(cmdows_problem.root.unknowns[xpath_to_param(x_t_fs)]),
        str(cmdows_problem.root.unknowns[xpath_to_param(x_t_rs)]),
        str(cmdows_problem.root.unknowns[xpath_to_param(x_t_ts)]),
        str(cmdows_problem.root.unknowns[xpath_to_param(x_t_bs)])))

    # Finally clean up the problem
    cmdows_problem.cleanup()
