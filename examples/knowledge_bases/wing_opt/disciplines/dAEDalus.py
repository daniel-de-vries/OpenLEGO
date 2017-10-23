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

This file contains the definitions of all the dAEDalus disciplines along with static variables they use.
"""
from __future__ import absolute_import, division, print_function

import abc
import os
import time

import matlab.engine
import numpy as np
from lxml import etree

from examples.knowledge_bases.wing_opt.disciplines.WingObjectModel import WingObjectModel
from examples.knowledge_bases.wing_opt.disciplines.xpaths import *
from openlego.api import AbstractDiscipline
from openlego.utils.general_utils import try_hard
from openlego.utils.xml_utils import xml_safe_create_element

dir_path = os.path.dirname(os.path.realpath(__file__))
_mles = {}

n_seg_x = 10
n_seg_y = 20


class LoadCaseSpecific(AbstractDiscipline):
    """Abstract base class storing the number of wing segments load cases as member properties in the constructor.

    Attributes
    ----------
        n_wing_segments : int
            Number of wing segments.

        n_load_cases : int
            Number of load cases.
    """

    def __init__(self, n_wing_segments=2, n_load_cases=1):
        # type: (int, int) -> None
        """Create an instance of the `LoadCollector` discipline.

        Parameters
        ----------
            n_wing_segments : int(2)
                Number of wing segments.

            n_load_cases : int(1)
                Number of load cases.
        """
        super(LoadCaseSpecific, self).__init__()
        self.n_wing_segments = n_wing_segments
        self.n_load_cases = n_load_cases

    @property
    def creator(self):
        return 'D. de Vries'

    @abc.abstractmethod
    def generate_input_xml(self):
        super(LoadCaseSpecific, self).generate_input_xml()

    @abc.abstractmethod
    def generate_output_xml(self):
        super(LoadCaseSpecific, self).generate_output_xml()

    @staticmethod
    @abc.abstractmethod
    def execute(in_file, out_file):
        super(LoadCaseSpecific, in_file).execute(in_file, out_file)

    @staticmethod
    def get_n_loadcases(tree):
        # type: (etree._ElementTree) -> int
        """ Obtain the number of load cases from the XML tree representing a CPACS file.

        Parameters
        ----------
            tree : :obj:`etree._ElementTree`
                `etree._ElementTree` corresponding to a CPACS file.

        Returns
        -------
            int
                Number of load cases defined in the CPACS file.
        """
        return len(tree.xpath('/'.join([x_loadcases, x_loadcase.split('/')[-1][:-4]])))


class SteadyModelInitializer(LoadCaseSpecific):
    """Initialization of the geometric and structural dAEDalus models using Matlab.

    This discipline takes a CPACS file with a fully defined wing as input and initializes the geometric and structural
    models of dAEDalus accordingly. The weight of the wing is calculated and stored in the output file, along with some
    pseudo-variables to aid linking this dicipline to the other dAEDalus disciplines.

    Behind the scenes, this discipline initializes the geometric and structural models and stores these in the workspace
    of the Matlab shared engine for each load case. Subsequent dAEDalus discipline calls can use these initialized
    models if they have the names of the Matlab shared engines.
    """

    MATLAB_TIMEOUT = 1800.

    def __init__(self, n_wing_segments=2, n_load_cases=1):
        super(SteadyModelInitializer, self).__init__(n_wing_segments, n_load_cases)

    @property
    def description(self):
        return 'dAEDalus Steady Model Initializer'

    def generate_input_xml(self):
        # type: () -> str
        """Input is a CPACS file with at least a fully defined wing.

        It is possible to specify a timeout for Matlab, the name of a Matlab sharedEngine, and the timestamp at which
        this engine was shared. However, the Matlab engine should normally be left under the control of this
        discipline.
        """
        wd = WingObjectModel(self.n_wing_segments)
        s = wd.generate_output_xml()

        parser = etree.XMLParser(remove_blank_text=True, encoding='utf-8')
        doc = etree.fromstring(s, parser)

        for i in range(1, self.n_load_cases + 1):
            xml_safe_create_element(doc, x_ml_timeout % i, self.MATLAB_TIMEOUT)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_output_xml(self):
        # type: () -> str
        """Output is a CPACS file containing a predefined number of load cases.

        Each load case will be assigned pseudo-variables allowing for subsequent disciplines to connect to this
        discipline (geometric and structural model), as well as the name of the shared Matlab engine and its timestamp
        of creation.
        """
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        xml_safe_create_element(doc, x_m_wing, 0.)

        for i in range(1, self.n_load_cases + 1):
            xml_safe_create_element(doc, x_geom % i, 1)
            xml_safe_create_element(doc, x_stru % i, 1)

            xml_safe_create_element(doc, x_ml_name % i, 'engine_name')
            xml_safe_create_element(doc, x_ml_timestamp % i, 0.)

            for j in range(3):
                xml_safe_create_element(doc, x_grid_initial[j] % i, np.zeros(2 * (n_seg_x + 1) * (n_seg_y + 1) * self.n_wing_segments))

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def execute(in_file, out_file):
        """Call the Matlab function dAEDalusSteadyModelInitializer() and store the resulting mass of the wingbox in the
        output XML file for each load case.

        The name of the Matlab engine is stored in CPACS. In this way it can be shared with all subsequent disciplines
        that need to use the same instance of Matlab in order to share the workspace.
        """
        doc_in = etree.parse(in_file)

        root = etree.Element('cpacs')
        doc_out = etree.ElementTree(root)

        for i in range(1, LoadCaseSpecific.get_n_loadcases(doc_in) + 1):
            timeout = SteadyModelInitializer.MATLAB_TIMEOUT
            elem_timeout = doc_in.xpath(x_ml_timeout % i)
            if len(elem_timeout):
                timeout = float(elem_timeout[0].text)

            timestamp = 0.
            elem_timestamp = doc_in.xpath(x_ml_timestamp % i)
            if len(elem_timestamp):
                timestamp = float(elem_timestamp[0].text)

            # Obtain the current matlab engine if it is still valid and exists
            engine_name = ''
            mle = None
            elem_ml_name = doc_in.xpath(x_ml_name % i)
            if len(elem_ml_name):
                engine_name = elem_ml_name[0].text
                if engine_name in _mles:
                    if time.time() - timestamp < timeout:
                        mle = _mles[engine_name]
                        mle.cd(dir_path)
                    else:
                        _mles.pop(engine_name)

            # If a matlab engine was not connected to, start a new one and reset the timestamp
            if mle is None:
                mle, engine_name, timestamp = SteadyModelInitializer.start_new_matlab_engine()
                mle.cd(dir_path)

            # Call the matlab function
            m_wing, initial_grid = mle.dAEDalusSteadyModelInitializer(
                in_file, matlab.double([n_seg_x]), matlab.double([n_seg_y]), nargout=2)
            initial_grid = np.array(initial_grid)

            # Finally write the output to the output xml file
            xml_safe_create_element(doc_out, x_m_wing, m_wing)
            xml_safe_create_element(doc_out, x_m_wing_copy, m_wing)

            for j in range(3):
                xml_safe_create_element(doc_out, x_grid_initial[j] % i, initial_grid[j, :])

            xml_safe_create_element(doc_out, x_geom % i, 1)
            xml_safe_create_element(doc_out, x_stru % i, 1)

            xml_safe_create_element(doc_out, x_ml_name % i, engine_name)
            xml_safe_create_element(doc_out, x_ml_timestamp % i, timestamp)

        doc_out.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def start_new_matlab_engine():
        """Ensure the Matlab engine is renewed once the MATLAB_TIMEOUT is expired.

        This function uses the try_hard() function from the framework.util module to ensure Matlab is started
        successfully when it needs to be.
        """
        mle = try_hard(matlab.engine.start_matlab, '-nodesktop -noslpash -nojvm')
        timestamp = time.time()
        mle.matlab.engine.shareEngine(nargout=0)
        engine_name = mle.matlab.engine.engineName(nargout=1)

        _mles.update({engine_name: mle})

        return mle, engine_name, timestamp


class SteadyAerodynamicModelInitializer(LoadCaseSpecific):
    """Initialization of the steady aerodynamic dAEDalus model.

    This discipline takes a CPACS file with a number of load cases containing a Mach number, altitude, and load factor.
    Furthermore, the pseudo-variable pointing to the geometric model of each load case, as well as the name of the load
    case's shared Matlab engine are required.

    Behind the scenes, the initialized geometric and structural models that were stored in the Matlab shared engine's
    workspace are used, and the aerodynamic model is stored there too once it is initialized.
    """

    def __init__(self, n_wing_segments=2, n_load_cases=1):
        super(SteadyAerodynamicModelInitializer, self).__init__(n_wing_segments, n_load_cases)

    @property
    def description(self):
        return 'dAEDalus Steady Aerodynamic Model Initializer'

    def generate_input_xml(self):
        # type: () -> str
        """Input is a CPACS file containing the Mach number, altitude, and load factor for each load case.

        The link to the geometric model and name of the Matlab shared engine for each load case is also required.
        """
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for i in range(1, self.n_load_cases + 1):
            xml_safe_create_element(doc, x_M % i, 0.)
            xml_safe_create_element(doc, x_H % i, 0.)
            xml_safe_create_element(doc, x_n % i, 0.)

            xml_safe_create_element(doc, x_geom % i, 1)
            xml_safe_create_element(doc, x_ml_name % i, 'engine_name')

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_output_xml(self):
        # type: () -> str
        """Output is a CPACS file containing the lift- and friction drag coefficient for each load case, as well as a
        link to the aerodynamic model.
        """
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for i in range(1, self.n_load_cases + 1):
            xml_safe_create_element(doc, x_CL % i, 0.)
            xml_safe_create_element(doc, x_CDf % i, 0.)

            xml_safe_create_element(doc, x_aero % i, 1)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def execute(in_file, out_file):
        """Call the Matlab function dAEDalusSteadyAerodynamicModelInitializer() and store the resulting values of C_L
        and C_D_f in the the CPACS output file.
        """
        doc_in = etree.parse(in_file)

        root = etree.Element('cpacs')
        doc_out = etree.ElementTree(root)

        for i in range(1, LoadCaseSpecific.get_n_loadcases(doc_in) + 1):
            M = float(doc_in.xpath(x_M % i)[0].text)
            H = float(doc_in.xpath(x_H % i)[0].text)
            n = float(doc_in.xpath(x_n % i)[0].text)

            engine_name = doc_in.xpath(x_ml_name % i)[0].text

            mle = _mles[engine_name]
            C_L, C_D_f = mle.dAEDalusSteadyAerodynamicModelInitializer(float(M), float(H), float(n), nargout=2)

            xml_safe_create_element(doc_out, x_CL % i, C_L)
            xml_safe_create_element(doc_out, x_CDf % i, C_D_f)
            xml_safe_create_element(doc_out, x_aero % i, 1)

        doc_out.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)


class SteadyAerodynamicAnalysis(LoadCaseSpecific):
    """"Steady aerodynamic analysis of dAEDalus using Matlab."""

    def __init__(self, n_wing_segments=2, n_load_cases=1):
        super(SteadyAerodynamicAnalysis, self).__init__(n_wing_segments, n_load_cases)
        self.previous_grids = n_load_cases * [None]

    @property
    def description(self):
        return 'dAEDalus Steady Aerodynamic Analysis'

    def generate_input_xml(self):
        # type: () -> str
        """Input is a CPACS file with the lift coefficient and deflected grid of the wing for each load case, as well
        as links to the geometric and aerodynamic models.
        """
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for i in range(1, self.n_load_cases + 1):
            xml_safe_create_element(doc, x_CL % i, 0.)

            xml_safe_create_element(doc, x_geom % i, 1)
            xml_safe_create_element(doc, x_aero % i, 1)
            xml_safe_create_element(doc, x_ml_name % i, 'engine_name')

            for j in range(3):
                xml_safe_create_element(doc, x_grid_initial[j] % i, np.zeros(2 * (n_seg_x + 1) * (n_seg_y + 1) * self.n_wing_segments))
                xml_safe_create_element(doc, x_grid[j] % i, np.zeros(2 * (n_seg_x + 1) * (n_seg_y + 1) * self.n_wing_segments))

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_output_xml(self):
        # type: () -> str
        """Output is a CPACS file with the induced drag coefficient and link to the aerodynamic forces."""
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for i in range(1, self.n_load_cases + 1):
            xml_safe_create_element(doc, x_CDi % i, 0.)

            for j in range(3):
                xml_safe_create_element(doc, x_grid_guess[j] % i, np.zeros(2 * (n_seg_x + 1) * (n_seg_y + 1) * self.n_wing_segments))

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def execute(in_file, out_file):
        """Call the Matlab function dAEDalusSteadyAerodynamicAnalysis() and store the resulting value of C_D_i and a
        link to the aerodynamic forces for each load case.
        """
        doc_in = etree.parse(in_file)

        root = etree.Element('cpacs')
        doc_out = etree.ElementTree(root)

        for i in range(1, LoadCaseSpecific.get_n_loadcases(doc_in) + 1):
            grid = 3 * [None]
            s = 0.
            for j in range(3):
                g = np.array(doc_in.xpath(x_grid[j] % i)[0].text.split(';'), dtype=float).tolist()
                s += np.sum(np.square(g))
                grid[j] = g
            if s == 0:
                for j in range(3):
                    grid[j] = np.array(doc_in.xpath(x_grid_initial[j] % i)[0].text.split(';'), dtype=float).tolist()

            C_L = float(doc_in.xpath(x_CL % i)[0].text)

            engine_name = doc_in.xpath(x_ml_name % i)[0].text

            mle = _mles[engine_name]
            C_D_i = mle.dAEDalusSteadyAerodynamicAnalysis(matlab.double(grid), float(C_L), nargout=1)

            xml_safe_create_element(doc_out, x_CDi % i, C_D_i)
            for j in range(3):
                xml_safe_create_element(doc_out, x_grid_guess[j] % i, grid[j])

        doc_out.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)


class SteadyStructuralAnalysis(LoadCaseSpecific):
    """Steady structural analysis of dAEDalus using Matlab."""

    def __init__(self, n_wing_segments=2, n_load_cases=1):
        super(SteadyStructuralAnalysis, self).__init__(n_wing_segments, n_load_cases)

    @property
    def description(self):
        return 'dAEDalus Steady Structural Analysis'

    def generate_input_xml(self):
        # type: () -> str
        """Input is a CPACS file containing the name of the Matlab shared engine, the links to all three models
        (geometric, structural, and aerodynamic), and the link the the aerodynamic forces for each load case.
        """
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for i in range(1, self.n_load_cases + 1):
            xml_safe_create_element(doc, x_ml_name % i, 'engine_name')
            xml_safe_create_element(doc, x_geom % i, 1)
            xml_safe_create_element(doc, x_stru % i, 1)
            xml_safe_create_element(doc, x_aero % i, 1)
            for j in range(3):
                xml_safe_create_element(doc, x_grid_guess[j] % i, np.zeros(2 * (n_seg_x + 1) * (n_seg_y + 1) * self.n_wing_segments))

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_output_xml(self):
        # type: () -> str
        """Output is a CPACS file with the stresses in the front/rear spars and in the top/bottom skins for each load
        case, as well as the deflected grid for each load case.
        """
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for i in range(1, self.n_load_cases + 1):
            for x_sigma in x_sigmas_in:
                xml_safe_create_element(doc, x_sigma % i, np.zeros(self.n_wing_segments))

            for j in range(3):
                xml_safe_create_element(doc, x_grid[j] % i, np.zeros(2 * (n_seg_x + 1) * (n_seg_y + 1) * self.n_wing_segments))

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def execute(in_file, out_file):
        """Call the Matlab function dAEDalusSteadyStructuralAnalysis() and store the resulting values of the stresses
        (sigma_*) and the deflected grid in the corresponding unknowns.
        """
        doc_in = etree.parse(in_file)

        root = etree.Element('cpacs')
        doc_out = etree.ElementTree(root)

        for i in range(1, LoadCaseSpecific.get_n_loadcases(doc_in) + 1):
            engine_name = doc_in.xpath(x_ml_name % i)[0].text
            mle = _mles[engine_name]
            sigma_fs, sigma_rs, sigma_ts, sigma_bs, deflected_grid = mle.dAEDalusSteadyStructuralAnalysis(nargout=5)

            sigmas = [sigma_fs, sigma_rs, sigma_ts, sigma_bs]
            for j in range(4):
                xml_safe_create_element(doc_out, x_sigmas_in[j] % i, np.array(sigmas[j]))

            for j in range(3):
                xml_safe_create_element(doc_out, x_grid[j] % i, np.array(deflected_grid[j]))

        doc_out.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)


class LoadCollector(LoadCaseSpecific):
    """Defines the Load Collector discipline.

    This discipline takes the maximum value of the stresses in the front/rear spars and top/bottom skins for any number
    of load cases and returns a single set of critical stresses.
    """

    def __init__(self, n_wing_segments=2, n_load_cases=1):
        super(LoadCollector, self).__init__(n_wing_segments, n_load_cases)

    @property
    def description(self):
        return 'Load Collector'

    def generate_input_xml(self):
        # type: () -> str
        """Input is a CPACS file with the stresses in the front/rear spars and in the top/bottom skins for each load
        case.
        """
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for i in range(1, self.n_load_cases + 1):
            for x_sigma in x_sigmas_in:
                xml_safe_create_element(doc, x_sigma % i, np.zeros(self.n_wing_segments))

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_output_xml(self):
        # type: () -> str
        """Output is a CPACS file containing the maximum stresses in the front/rear spars and in the top/bottom skins
        across all load cases.
        """
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for x_sigma in x_sigmas_out:
            xml_safe_create_element(doc, x_sigma, np.zeros(self.n_wing_segments))

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def execute(in_file, out_file='LC-output-loc.xml'):
        """Computes the maximum stresses across all load cases."""
        doc_in = etree.parse(in_file)

        sigmas = 4 * [np.ndarray((0,))]
        for i in range(1, LoadCaseSpecific.get_n_loadcases(doc_in) + 1):
            for j in range(4):
                data = np.array(doc_in.xpath(x_sigmas_in[j] % i)[0].text.split(';'), dtype=float)
                if i == 1:
                    sigmas[j] = data
                else:
                    sigmas[j] = np.maximum(sigmas[j], data)

        # Write results to output XML file
        root = etree.Element('cpacs')
        doc_out = etree.ElementTree(root)
        for i in range(4):
            xml_safe_create_element(doc_out, x_sigmas_out[i], sigmas[i])

        doc_out.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)


class SteadyLiftDistribution(LoadCaseSpecific):
    """Calculation of the lift distribution using Matlab."""

    def __init__(self, n_wing_segments=2, n_load_cases=1):
        super(SteadyLiftDistribution, self).__init__(n_wing_segments, n_load_cases)

    @property
    def description(self):
        return 'Steady Lift Distribution'

    def generate_input_xml(self):
        # type: () -> str
        """Input is a CPACS file with the name of the Matlab shared engine and links to the geometric model, aerodynamic
        model and aerodynamic forces for each load case.
        """
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for i in range(1, self.n_load_cases + 1):
            xml_safe_create_element(doc, x_ml_name % i, 'engine_name')
            xml_safe_create_element(doc, x_geom % i, 1)
            xml_safe_create_element(doc, x_aero % i, 1)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_output_xml(self):
        # type: () -> str
        """Output is a CPACS file with the normalized y-coordinates and corresponding normalized section lift forces for
        each load case.
        """
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for i in range(1, self.n_load_cases + 1):
            xml_safe_create_element(doc, x_y_norm % i, np.zeros(n_seg_y * self.n_wing_segments))
            xml_safe_create_element(doc, x_l_norm % i, np.zeros(n_seg_y * self.n_wing_segments))

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def execute(in_file, out_file):
        """Call the Matlab function dAEDalusSteadyLiftDistribution() and store the resulting values of y_norm
        (normalized y-coordinates) and l_norm (normalized lift) in the output XML.
        """
        doc_in = etree.parse(in_file)

        root = etree.Element('cpacs')
        doc_out = etree.ElementTree(root)

        for i in range(1, LoadCaseSpecific.get_n_loadcases(doc_in) + 1):
            engine_name = doc_in.xpath(x_ml_name % i)[0].text
            mle = _mles[engine_name]
            y_norm, l_norm = mle.dAEDalusSteadyLiftDistribution(nargout=2)

            xml_safe_create_element(doc_out, x_y_norm % i, np.array(y_norm))
            xml_safe_create_element(doc_out, x_l_norm % i, np.array(l_norm))

        doc_out.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)
