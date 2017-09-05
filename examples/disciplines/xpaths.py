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

This file contains all the XPaths and utility string constants used by the dAEDalus disciplines.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


sigma_names = ['sigma_fs', 'sigma_rs', 'sigma_ts', 'sigma_bs']

""" CPACS """
x_vehicles = '/cpacs/vehicles'
x_model = x_vehicles + r"/aircraft/model[@uID='model']"
x_ref = '/'.join([x_model, 'reference'])
x_wing = '/'.join([x_model, r"wings/wing[@uID='wing'][@symmetry='x-z-plane']"])
x_sec = '/'.join([x_wing, r"sections/section[@uID='sec_%d']"])
x_elem = '/'.join([x_sec, r"elements/element[@uID='elem_%d']"])
x_mbd = '/'.join([x_model, 'analyses/massBreakdown'])
x_global = '/'.join([x_model, 'global'])
x_perf = '/'.join([x_global, 'performanceTargets'])

x_ref_area = '/'.join([x_ref, 'area'])
x_ref_length = '/'.join([x_ref, 'length'])
x_compseg = '/'.join([x_wing, r"componentSegments/componentSegment[@uID='compSeg_%d']"])
x_struct = '/'.join([x_compseg, 'structure'])
x_sparposs = '/'.join([x_struct, 'spars/sparPositions'])
x_fs_r_xsi = '/'.join([x_sparposs, r"sparPosition[@uID='fs_%d_r']/xsi"])
x_fs_t_xsi = '/'.join([x_sparposs, r"sparPosition[@uID='fs_%d_t']/xsi"])
x_rs_r_xsi = '/'.join([x_sparposs, r"sparPosition[@uID='rs_%d_r']/xsi"])
x_rs_t_xsi = '/'.join([x_sparposs, r"sparPosition[@uID='rs_%d_t']/xsi"])
x_sparsegs = '/'.join([x_struct, 'spars/sparSegments'])
x_fs_web_t = '/'.join([x_sparsegs, r"sparSegment[@uID='fs_%d']/sparCrossSection/web1/material/thickness"])
x_fs_lowerCap_t = '/'.join([x_sparsegs, r"sparSegment[@uID='fs_%d']/sparCrossSection/lowerCap/material/thickness"])
x_fs_upperCap_t = '/'.join([x_sparsegs, r"sparSegment[@uID='fs_%d']/sparCrossSection/upperCap/material/thickness"])
x_rs_web_t = '/'.join([x_sparsegs, r"sparSegment[@uID='rs_%d']/sparCrossSection/web1/material/thickness"])
x_rs_lowerCap_t = '/'.join([x_sparsegs, r"sparSegment[@uID='rs_%d']/sparCrossSection/lowerCap/material/thickness"])
x_rs_upperCap_t = '/'.join([x_sparsegs, r"sparSegment[@uID='rs_%d']/sparCrossSection/upperCap/material/thickness"])
x_mSkins = '/'.join([x_model, r"analyses/massBreakdown/mOEM/mEM/mStructure/mWingsStructure/"
                            r"mWingStructure/mComponentSegment[%d]/mWingBox"])

""" Wing optimization problem """
x_opt = '/cpacs/toolspecific/wingOptimizationProblem'
x_planform = '/'.join([x_opt, 'planform'])
x_structure = '/'.join([x_opt, 'structure'])
x_reference = '/'.join([x_opt, 'reference'])

x_const = '/'.join([x_opt, 'constants'])
x_con = '/'.join([x_opt, 'constraints'])
x_obj = '/'.join([x_opt, 'objectives'])

x_c = '/'.join([x_planform, 'c'])
x_tc = '/'.join([x_planform, 'tc'])
x_epsilon = '/'.join([x_planform, 'epsilon'])
x_b = '/'.join([x_planform, 'b'])
x_Lambda = '/'.join([x_planform, 'Lambda'])
x_Gamma = '/'.join([x_planform, 'Gamma'])
x_incidence = '/'.join([x_planform, 'incidence'])

x_xsi_fs = '/'.join([x_structure, 'xsi_fs'])
x_xsi_rs = '/'.join([x_structure, 'xsi_rs'])
x_t_fs = '/'.join([x_structure, 't_fs'])
x_t_rs = '/'.join([x_structure, 't_rs'])
x_t_ts = '/'.join([x_structure, 't_ts'])
x_t_bs = '/'.join([x_structure, 't_bs'])
x_t_skin = '/'.join([x_structure, 't_skin'])

x_m_fixed = '/'.join([x_reference, 'm_fixed'])
x_m_payload = '/'.join([x_reference, 'm_payload'])
x_f_m_sys = '/'.join([x_reference, 'f_m_sys'])
x_f_m_wings = '/'.join([x_reference, 'f_m_wings'])
x_m_mlw = '/'.join([x_reference, 'm_MLW'])

x_SFC = '/'.join([x_reference, 'SFC'])
x_m_fuel_res = '/'.join([x_reference, 'm_fuel_res'])
x_CDfus = '/'.join([x_reference, 'C_D_fus'])
x_CDother = '/'.join([x_reference, 'C_D_other'])
x_R = '/'.join([x_reference, 'R'])

x_rho_skin = '/'.join([x_reference, 'rho_skin'])
x_sigma_yield = '/'.join([x_reference, 'sigma_yield'])
x_S_ref_init = '/'.join([x_reference, 'S_ref_init'])
x_CL_buffet = '/'.join([x_reference, 'C_L_buffet'])
x_m_wing_init = '/'.join([x_reference, 'm_wing_init'])
x_m_fuel_init = '/'.join([x_reference, 'm_fuel_init'])

x_m_fuel_copy = '/'.join([x_reference, 'm_fuel'])
x_m_wing_copy = '/'.join([x_reference, 'm_wing'])

x_con_sigmas = ['/'.join([x_con, 'con_' + sigma]) for sigma in sigma_names]
x_con_exposed_area = '/'.join([x_con, 'con_exposed_area'])
x_con_buffet = '/'.join([x_con, 'con_buffet'])

x_obj_m_fuel = '/'.join([x_obj, 'obj_m_fuel'])
x_obj_m_wing = '/'.join([x_obj, 'obj_m_wing'])


""" dAEDalus """
x_dAE = '/cpacs/toolspecific/dAEDalus'
x_m_wing = '/'.join([x_dAE, 'm_wing'])

x_loadcases = '/'.join([x_dAE, 'loadCases'])
x_loadcase = '/'.join([x_loadcases, 'loadCase[%d]'])

x_M = '/'.join([x_loadcase, 'M'])
x_H = '/'.join([x_loadcase, 'H'])
x_n = '/'.join([x_loadcase, 'n'])
x_CL = '/'.join([x_loadcase, 'C_L'])
x_CDf = '/'.join([x_loadcase, 'C_D_f'])
x_CDi = '/'.join([x_loadcase, 'C_D_i'])

x_grid_initial = ['/'.join([x_loadcase, 'initial_grid/' + component]) for component in ['x', 'y', 'z']]
x_grid = ['/'.join([x_loadcase, 'deflected_grid/' + component]) for component in ['x', 'y', 'z']]
x_grid_guess = ['/'.join([x_loadcase, 'guess_grid/' + component]) for component in ['x', 'y', 'z']]

x_sigmas_in = ['/'.join([x_loadcase, sigma]) for sigma in sigma_names]
x_load_collector = '/'.join([x_dAE, 'load_collector'])
x_sigmas_out = ['/'.join([x_load_collector, sigma]) for sigma in sigma_names]

x_y_norm = '/'.join([x_loadcase, 'y_norm'])
x_l_norm = '/'.join([x_loadcase, 'l_norm'])

x_geom = '/'.join([x_loadcase, 'geometric_model'])
x_stru = '/'.join([x_loadcase, 'structural_model'])
x_aero = '/'.join([x_loadcase, 'aerodynamic_model'])

x_mle = '/'.join([x_loadcase, 'matlab_engine'])
x_ml_timeout = '/'.join([x_mle, 'timeout'])
x_ml_name = '/'.join([x_mle, 'name'])
x_ml_timestamp = '/'.join([x_mle, 'timestamp'])

""" FWE """
x_fwe = '/cpacs/toolspecific/fuel_weight_estimator'

x_CD = '/'.join([x_fwe, 'C_D'])
x_LD = '/'.join([x_fwe, 'L_D'])
x_m_fuel = '/'.join([x_fwe, 'm_fuel'])
x_fwe_CL = '/'.join([x_fwe, 'C_L'])
