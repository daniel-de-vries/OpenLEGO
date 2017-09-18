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

This file contains the definition of the ProblemDefinition discipline."""
from __future__ import absolute_import, division, print_function

import numpy as np
from lxml import etree

from examples.kb.kb_wing_opt.disciplines.xpaths import *
from openlego.discipline import AbstractDiscipline
from openlego.xml import xml_safe_create_element


class Constraints(AbstractDiscipline):
    """Defines all the constraints for the problem."""

    def __init__(self, n_wing_segments=2):
        # type: (int) -> None
        """Initialize the Constraints discipline for a given number of wing segments.

        Parameters
        ----------
            n_wing_segments : int
                Number of wing segments.
        """
        self.n_wing_segments = n_wing_segments
        super(Constraints, self).__init__()

    @property
    def creator(self):
        return 'D. de Vries'

    @property
    def description(self):
        return 'Calculates the constraint values for the wing optimization problem'

    def generate_input_xml(self):
        # type: () -> str
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        xml_safe_create_element(doc, x_sigma_yield, 1.)
        xml_safe_create_element(doc, x_S_ref_init, 1.)
        xml_safe_create_element(doc, x_CL_buffet, 1.)

        for x_sigma in x_sigmas_out:
            xml_safe_create_element(doc, x_sigma, np.zeros(self.n_wing_segments))

        xml_safe_create_element(doc, x_ref_area, 0.)
        xml_safe_create_element(doc, x_fwe_CL, 0.)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_output_xml(self):
        # type: () -> str
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        for x_sigma in x_con_sigmas:
            xml_safe_create_element(doc, x_sigma, np.zeros(self.n_wing_segments))
        # xml_safe_create_element(doc, x_con_ks, 0.)

        xml_safe_create_element(doc, x_con_exposed_area, 0.)
        xml_safe_create_element(doc, x_con_buffet, 0.)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def execute(in_file, out_file):
        doc_in = etree.parse(in_file)

        sigma_yield = float(doc_in.xpath(x_sigma_yield)[0].text)

        root = etree.Element('cpacs')
        doc_out = etree.ElementTree(root)
        for i in range(4):
            xml_safe_create_element(
                doc_out, x_con_sigmas[i],
                np.array(doc_in.xpath(x_sigmas_out[i])[0].text.split(';'), dtype=float)/sigma_yield - 1.)

        xml_safe_create_element(doc_out, x_con_exposed_area,
                                float(doc_in.xpath(x_ref_area)[0].text)/float(doc_in.xpath(x_S_ref_init)[0].text) - 1.)
        xml_safe_create_element(doc_out, x_con_buffet,
                                float(doc_in.xpath(x_fwe_CL)[0].text)/float(doc_in.xpath(x_CL_buffet)[0].text) - 1.)

        doc_out.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)


class Objectives(AbstractDiscipline):
    """Defines the objective functions for the problem."""

    def __init__(self):
        # type: (int) -> None
        """Initialize the Objectives discipline."""
        pass

    @property
    def creator(self):
        return 'D. de Vries'

    @property
    def description(self):
        return 'Calculates the objective values for the wing optimization problem'

    def generate_input_xml(self):
        # type: () -> str
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        xml_safe_create_element(doc, x_m_fuel_init, 1.)
        xml_safe_create_element(doc, x_m_fuel, 1.)
        xml_safe_create_element(doc, x_m_wing_init, 1.)
        xml_safe_create_element(doc, x_m_wing, 1.)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_output_xml(self):
        # type: () -> str
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        xml_safe_create_element(doc, x_obj_m_fuel, 0.)
        xml_safe_create_element(doc, x_obj_m_wing, 0.)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def execute(in_file, out_file):
        doc_in = etree.parse(in_file)

        root = etree.Element('cpacs')
        doc_out = etree.ElementTree(root)

        xml_safe_create_element(doc_out, x_obj_m_fuel,
                                float(doc_in.xpath(x_m_fuel)[0].text) / float(doc_in.xpath(x_m_fuel_init)[0].text))
        xml_safe_create_element(doc_out, x_obj_m_wing, 
                                float(doc_in.xpath(x_m_wing)[0].text) / float(doc_in.xpath(x_m_wing_init)[0].text))

        doc_out.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)
