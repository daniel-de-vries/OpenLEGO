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

This file contains the definition of the FuelWeightEstimator class along with static variables it uses.
"""
from __future__ import absolute_import, division, print_function

import os

from openlego.test_suite.test_examples.wing_opt.kb.disciplines.dAEDalus import LoadCaseSpecific
from lxml import etree
from numpy import sqrt, expm1, exp

from openlego.test_suite.test_examples.wing_opt.kb.disciplines.xpaths import *
from openlego.utils.xml_utils import xml_safe_create_element
from openlego.partials.partials import Partials

dir_path = os.path.dirname(os.path.realpath(__file__))

g0 = 9.80665        # Standard gravitational acceleration, [m/s^2]
rho0 = 1.225        # ISA sea level density, [kg/m^3]
R_air = 287.05287   # ISA specific gas constant, [J/kg/K]
P0 = 101325         # ISA sea level pressure, [Pa]
T0 = 288.15         # ISA sea level temperature, [K]
beta = -0.0065      # ISA temperature lapse rate, [K/m]
kappa = 1.4         # ISA ratio of specific heats, [-]


class FuelWeightEstimator(LoadCaseSpecific):
    """Definition of the Fuel Weight Estimator discipline.

    This discipline estimates the weight of the fuel required to fly a given mission based on the Breguet range
    equation.
    """

    def __init__(self, n_wing_segments=2, n_load_cases=1):
        super(FuelWeightEstimator, self).__init__(n_wing_segments, n_load_cases)

    @property
    def creator(self):
        return 'D. de Vries'

    @property
    def description(self):
        return 'Fuel weight estimator'

    def generate_input_xml(self):
        # type: () -> str
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for i in range(1, self.n_load_cases + 1):
            xml_safe_create_element(doc, x_M % i, 0)
            xml_safe_create_element(doc, x_H % i, 0)
            xml_safe_create_element(doc, x_n % i, 0)
            xml_safe_create_element(doc, x_CDf % i, 0)
            xml_safe_create_element(doc, x_CDi % i, 0)
            xml_safe_create_element(doc, x_CL % i, 0)

        xml_safe_create_element(doc, x_CDfus, 0)
        xml_safe_create_element(doc, x_CDother, 0)
        xml_safe_create_element(doc, x_R, 0)
        xml_safe_create_element(doc, x_SFC, 0)
        xml_safe_create_element(doc, x_m_fuel_res, 0)
        xml_safe_create_element(doc, x_m_fixed, 0)
        xml_safe_create_element(doc, x_m_wing, 0)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_output_xml(self):
        # type: () -> str
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        xml_safe_create_element(doc, x_CD, 0)
        xml_safe_create_element(doc, x_LD, 0)
        xml_safe_create_element(doc, x_fwe_CL, 0)
        xml_safe_create_element(doc, x_m_fuel, 0)
        xml_safe_create_element(doc, x_m_mtow, 0)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_partials_xml(self):
        partials = Partials()
        partials.declare_partials(x_CD, [x_CDi % 1, x_CDf % 1, x_CDfus, x_CDother])
        partials.declare_partials(x_LD, [x_CDi % 1, x_CDf % 1, x_CDfus, x_CDother, x_CL % 1])
        partials.declare_partials(x_fwe_CL, x_CL % 1)
        partials.declare_partials(x_m_fuel, [x_CDi % 1, x_CDf % 1, x_CDfus, x_CDother, x_CL % 1,
                                             x_M % 1, x_H % 1, x_R, x_SFC, x_m_fuel_res, x_m_fixed, x_m_wing])
        partials.declare_partials(x_m_mtow, [x_CDi % 1, x_CDf % 1, x_CDfus, x_CDother, x_CL % 1,
                                             x_M % 1, x_H % 1, x_R, x_SFC, x_m_fuel_res, x_m_fixed, x_m_wing])
        return partials.get_string()

    @property
    def supplies_partials(self):
        return False

    @staticmethod
    def execute(in_file, out_file='FWE-output-loc.xml'):
        # Obtain data from XML file
        tree = etree.parse(in_file)

        M = H = C_D_f = C_D_i = C_L = 0
        for i in range(1, LoadCaseSpecific.get_n_loadcases(tree) + 1):
            M = float(tree.xpath(x_M % i)[0].text)                      # Mach number, [-]
            H = float(tree.xpath(x_H % i)[0].text)                      # Altitude, [m]
            C_D_f = float(tree.xpath(x_CDf % i)[0].text)                # Friction drag coefficient, [-]
            C_D_i = float(tree.xpath(x_CDi % i)[0].text)                # Induced drag coefficient, [-]
            C_L = float(tree.xpath(x_CL % i)[0].text)                   # Lift coefficient, [-]

            if round(float(tree.xpath(x_n % i)[0].text)) == 1.:
                break

        C_D_fus = float(tree.xpath(x_CDfus)[0].text)            # Fuselage drag coefficient, [-]
        C_D_other = float(tree.xpath(x_CDother)[0].text)        # Remaining drag coefficient, [-]
        R = float(tree.xpath(x_R)[0].text)                      # Range, [m]
        SFC = float(tree.xpath(x_SFC)[0].text)                  # Specific fuel consumption, [kg/N/s]
        m_fuel_res = float(tree.xpath(x_m_fuel_res)[0].text)    # Reserve fuel weight, [kg]
        m_fixed = float(tree.xpath(x_m_fixed)[0].text)          # Fixed weight, [kg]
        m_wing = float(tree.xpath(x_m_wing)[0].text)            # Wing weight, [kg]

        # Perform calculations
        C_D = C_D_i + C_D_f + C_D_fus + C_D_other               # Total drag coefficient, [-]
        L_D = C_L/C_D                                           # Aerodynamic efficiency, L/D, [-]
        LW = m_fixed + m_wing + m_fuel_res                      # Landing weight, [kg]
        T = T0 + beta * H                                       # Temperature at altitude, [K]
        a = sqrt(kappa * R_air * T)                             # Speed of sound at altitude, [m/s]

        if L_D == 0:
            m_fuel = 0.
        else:
            m_fuel = LW * expm1(R * g0 * SFC / (a * M) / L_D)       # Required fuel weight, [kg]

        m_mtow = m_fuel + LW
        m_fuel = m_fuel + m_fuel_res

        # Write results to output XML file
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)
        xml_safe_create_element(doc, x_CD, C_D)
        xml_safe_create_element(doc, x_LD, L_D)
        xml_safe_create_element(doc, x_fwe_CL, C_L)
        xml_safe_create_element(doc, x_m_fuel, m_fuel)
        xml_safe_create_element(doc, x_m_mtow, m_mtow)
        doc.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def linearize(in_file, partials_file):
        tree = etree.parse(in_file)

        M = H = C_D_f = C_D_i = C_L = 0
        for i in range(1, LoadCaseSpecific.get_n_loadcases(tree) + 1):
            M = float(tree.xpath(x_M % i)[0].text)  # Mach number, [-]
            H = float(tree.xpath(x_H % i)[0].text)  # Altitude, [m]
            C_D_f = float(tree.xpath(x_CDf % i)[0].text)  # Friction drag coefficient, [-]
            C_D_i = float(tree.xpath(x_CDi % i)[0].text)  # Induced drag coefficient, [-]
            C_L = float(tree.xpath(x_CL % i)[0].text)  # Lift coefficient, [-]

            if round(float(tree.xpath(x_n % i)[0].text)) == 1.:
                break

        C_D_fus = float(tree.xpath(x_CDfus)[0].text)  # Fuselage drag coefficient, [-]
        C_D_other = float(tree.xpath(x_CDother)[0].text)  # Remaining drag coefficient, [-]
        R = float(tree.xpath(x_R)[0].text)  # Range, [m]
        SFC = float(tree.xpath(x_SFC)[0].text)  # Specific fuel consumption, [kg/N/s]
        m_fuel_res = float(tree.xpath(x_m_fuel_res)[0].text)  # Reserve fuel weight, [kg]
        m_fixed = float(tree.xpath(x_m_fixed)[0].text)  # Fixed weight, [kg]
        m_wing = float(tree.xpath(x_m_wing)[0].text)  # Wing weight, [kg]

        C_D = C_D_i + C_D_f + C_D_fus + C_D_other               # Total drag coefficient, [-]
        L_D = C_L / C_D  # Aerodynamic efficiency, L/D, [-]
        LW = m_fixed + m_wing + m_fuel_res  # Landing weight, [kg]
        T = T0 + beta * H  # Temperature at altitude, [K]
        a = sqrt(kappa * R_air * T)  # Speed of sound at altitude, [m/s]

        if L_D == 0:
            dLD_dd = 5*[1.]
            dmfuel_d2 = 12*[1.]
            dmmtow_dd = 12*[1.]
        else:
            m_fuel = LW * expm1(R * g0 * SFC / (a * M) / L_D)       # Required fuel weight, [kg]

            dmfuel_dd = LW*R*g0*SFC/(a*M*L_D)*exp(R*g0*SFC/(a*M*L_D))
            da_dH = .5*kappa*R_air/a

            dLD_dd = 4*[1./C_L] + [-C_D/C_L**2]
            dmfuel_d2 = 4*[dmfuel_dd/C_D] + [-dmfuel_dd/C_L, -dmfuel_dd/M,
                                             -dmfuel_dd/a*da_dH, dmfuel_dd/R,
                                             dmfuel_dd/SFC, 1., m_fuel/m_fuel_res,
                                             m_fuel/m_fixed, m_fuel/m_wing]
            dmmtow_dd = 4 * [dmfuel_dd / C_D] + [-dmfuel_dd / C_L, -dmfuel_dd / M,
                                                 -dmfuel_dd / a * da_dH, dmfuel_dd / R,
                                                 dmfuel_dd / SFC, 1., dmfuel_dd / m_fuel_res + 1.,
                                                 m_fuel / m_fixed + 1., m_fuel / m_wing + 1.]

        partials = Partials()
        partials.declare_partials(x_CD, [x_CDi % 1, x_CDf % 1, x_CDfus, x_CDother], 4*[1.])
        partials.declare_partials(x_LD, [x_CDi % 1, x_CDf % 1, x_CDfus, x_CDother, x_CL % 1], dLD_dd)
        partials.declare_partials(x_fwe_CL, x_CL % 1, 1.)
        partials.declare_partials(x_m_fuel, [x_CDi % 1, x_CDf % 1, x_CDfus, x_CDother, x_CL % 1,
                                             x_M % 1, x_H % 1, x_R, x_SFC, x_m_fuel_res, x_m_fixed, x_m_wing],
                                  dmfuel_d2)
        partials.declare_partials(x_m_mtow, [x_CDi % 1, x_CDf % 1, x_CDfus, x_CDother, x_CL % 1,
                                             x_M % 1, x_H % 1, x_R, x_SFC, x_m_fuel_res, x_m_fixed, x_m_wing],
                                  dmmtow_dd)

        partials.write(partials_file)