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

This file contains the definition of the Sellar D1 discipline.
"""
from __future__ import absolute_import, division, print_function

from lxml import etree

from openlego.api import AbstractDiscipline
from openlego.utils.xml_utils import xml_safe_create_element
from openlego.test_suite.test_examples.sellar_competences.kb import root_tag, x_x1, x_z1, x_z2, x_y2, x_y1
from openlego.partials.partials import Partials


class D1(AbstractDiscipline):

    @property
    def creator(self):
        return u'D. de Vries'

    @property
    def description(self):
        return u'First discipline of the Sellar problem'

    @property
    def supplies_partials(self):
        return True

    def generate_input_xml(self):
        root = etree.Element(root_tag)
        doc = etree.ElementTree(root)

        xml_safe_create_element(doc, x_x1, 0.)
        xml_safe_create_element(doc, x_z1, 0.)
        xml_safe_create_element(doc, x_z2, 0.)
        xml_safe_create_element(doc, x_y2, 0.)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_output_xml(self):
        root = etree.Element(root_tag)
        doc = etree.ElementTree(root)

        xml_safe_create_element(doc, x_y1, 0.)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_partials_xml(self):
        partials = Partials()
        partials.declare_partials(x_y1, [x_x1, x_y2, x_z1, x_z2])
        return partials.get_string()

    @staticmethod
    def execute(in_file, out_file):
        doc = etree.parse(in_file)
        z1 = float(doc.xpath(x_z1)[0].text)
        z2 = float(doc.xpath(x_z2)[0].text)
        x1 = float(doc.xpath(x_x1)[0].text)
        y2 = float(doc.xpath(x_y2)[0].text)

        y1 = z1**2. + x1 + z2 - .2*y2

        root = etree.Element(root_tag)
        doc = etree.ElementTree(root)
        xml_safe_create_element(doc, x_y1, y1)
        doc.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def linearize(in_file, partials_file):
        doc = etree.parse(in_file)
        z1 = float(doc.xpath(x_z1)[0].text)

        partials = Partials()
        partials.declare_partials(x_y1, [x_x1, x_y2, x_z1, x_z2], [1., -.2, 2.*z1, 1.])
        partials.write(partials_file)
