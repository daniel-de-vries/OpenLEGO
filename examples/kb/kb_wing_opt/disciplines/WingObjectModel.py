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

This file contains the definition of the WingObjectModel class along with a collection of static variables and methods
it uses.
"""
from __future__ import absolute_import, division, print_function

import os

import numpy as np
from lxml import etree
from typing import Optional, Union, Tuple, Sized

from openlego.discipline import AbstractDiscipline
from openlego.components import xml_safe_create_element

from examples.kb.kb_wing_opt.disciplines.xpaths import *

dir_path = os.path.dirname(os.path.realpath(__file__))

af_x = (1.000000, 0.997840, 0.995410, 0.992720, 0.989770, 0.986550, 0.983080, 0.979350, 0.975360, 0.971120,
        0.966630, 0.961890, 0.956900, 0.951680, 0.946210, 0.940500, 0.934550, 0.928380, 0.921980, 0.915350,
        0.908500, 0.901440, 0.894160, 0.886670, 0.878980, 0.871080, 0.862990, 0.854700, 0.846230, 0.837570,
        0.828730, 0.819720, 0.810540, 0.801190, 0.791680, 0.782020, 0.772210, 0.762260, 0.752160, 0.741940,
        0.731580, 0.721100, 0.710510, 0.699800, 0.688990, 0.678080, 0.667080, 0.655990, 0.644810, 0.633560,
        0.622240, 0.610860, 0.599420, 0.587930, 0.576390, 0.564820, 0.553210, 0.541580, 0.529920, 0.518250,
        0.506580, 0.494900, 0.483220, 0.471560, 0.459920, 0.448290, 0.436700, 0.425150, 0.413630, 0.402170,
        0.390760, 0.379410, 0.368130, 0.356920, 0.345790, 0.334740, 0.323780, 0.312930, 0.302170, 0.291520,
        0.280990, 0.270580, 0.260290, 0.250130, 0.240110, 0.230230, 0.220500, 0.210920, 0.201490, 0.192230,
        0.183140, 0.174220, 0.165480, 0.156920, 0.148540, 0.140360, 0.132370, 0.124580, 0.116990, 0.109610,
        0.102450, 0.095500, 0.088770, 0.082260, 0.075980, 0.069920, 0.064110, 0.058520, 0.053180, 0.048080,
        0.043220, 0.038610, 0.034260, 0.030150, 0.026300, 0.022700, 0.019370, 0.016290, 0.013480, 0.010920,
        0.008640, 0.006620, 0.004870, 0.003380, 0.002160, 0.001220, 0.000540, 0.000140, 0.000000, 0.000000,
        0.000140, 0.000540, 0.001220, 0.002160, 0.003380, 0.004870, 0.006620, 0.008640, 0.010920, 0.013480,
        0.016290, 0.019370, 0.022700, 0.026300, 0.030150, 0.034260, 0.038610, 0.043220, 0.048080, 0.053180,
        0.058520, 0.064110, 0.069920, 0.075980, 0.082260, 0.088770, 0.095500, 0.102450, 0.109610, 0.116990,
        0.124580, 0.132370, 0.140360, 0.148540, 0.156920, 0.165480, 0.174220, 0.183140, 0.192230, 0.201490,
        0.210920, 0.220500, 0.230230, 0.240110, 0.250130, 0.260290, 0.270580, 0.280990, 0.291520, 0.302170,
        0.312930, 0.323780, 0.334740, 0.345790, 0.356920, 0.368130, 0.379410, 0.390760, 0.402170, 0.413630,
        0.425150, 0.436700, 0.448290, 0.459920, 0.471560, 0.483220, 0.494900, 0.506580, 0.518250, 0.529920,
        0.541580, 0.553210, 0.564820, 0.576390, 0.587930, 0.599420, 0.610860, 0.622240, 0.633560, 0.644810,
        0.655990, 0.667080, 0.678080, 0.688990, 0.699800, 0.710510, 0.721100, 0.731580, 0.741940, 0.752160,
        0.762260, 0.772210, 0.782020, 0.791680, 0.801190, 0.810540, 0.819720, 0.828730, 0.837570, 0.846230,
        0.854700, 0.862990, 0.871080, 0.878980, 0.886670, 0.894160, 0.901440, 0.908500, 0.915350, 0.921980,
        0.928380, 0.934550, 0.940500, 0.946210, 0.951680, 0.956900, 0.961890, 0.966630, 0.971120, 0.975360,
        0.979350, 0.983080, 0.986550, 0.989770, 0.992720, 0.995410, 0.997840, 1.000000)
af_z = (0.000000, 0.001150, 0.002320, 0.003500, 0.004690, 0.005900, 0.007110, 0.008340, 0.009560, 0.010810,
        0.012040, 0.013270, 0.014490, 0.015680, 0.016880, 0.018070, 0.019240, 0.020410, 0.021570, 0.022730,
        0.023890, 0.025070, 0.026270, 0.027480, 0.028720, 0.029980, 0.031270, 0.032590, 0.033930, 0.035310,
        0.036690, 0.038090, 0.039510, 0.040920, 0.042320, 0.043720, 0.045100, 0.046460, 0.047790, 0.049090,
        0.050340, 0.051550, 0.052720, 0.053830, 0.054890, 0.055900, 0.056850, 0.057750, 0.058580, 0.059360,
        0.060080, 0.060760, 0.061380, 0.061950, 0.062470, 0.062940, 0.063360, 0.063720, 0.064070, 0.064350,
        0.064590, 0.064790, 0.064940, 0.065060, 0.065120, 0.065140, 0.065130, 0.065060, 0.064960, 0.064810,
        0.064610, 0.064370, 0.064090, 0.063750, 0.063380, 0.062970, 0.062510, 0.062010, 0.061460, 0.060890,
        0.060280, 0.059620, 0.058930, 0.058200, 0.057460, 0.056670, 0.055870, 0.055030, 0.054160, 0.053270,
        0.052350, 0.051400, 0.050450, 0.049450, 0.048440, 0.047390, 0.046340, 0.045250, 0.044150, 0.043020,
        0.041870, 0.040710, 0.039510, 0.038320, 0.037080, 0.035840, 0.034570, 0.033290, 0.031980, 0.030650,
        0.029290, 0.027920, 0.026510, 0.025090, 0.023630, 0.022130, 0.020610, 0.019050, 0.017450, 0.015840,
        0.014170, 0.012480, 0.010770, 0.009020, 0.007240, 0.005460, 0.003660, 0.001820, 0.000000, 0.000000,
        -0.001550, -0.003060, -0.004570, -0.006050, -0.007490, -0.008930, -0.010320, -0.011680, -0.013020,
        -0.014310, -0.015580, -0.016810, -0.018010, -0.019190, -0.020340, -0.021460, -0.022570, -0.023650,
        -0.024730, -0.025770, -0.026820, -0.027850, -0.028860, -0.029860, -0.030860, -0.031830, -0.032810,
        -0.033760, -0.034700, -0.035620, -0.036540, -0.037430, -0.038290, -0.039130, -0.039960, -0.040770,
        -0.041540, -0.042290, -0.043010, -0.043710, -0.044370, -0.045000, -0.045600, -0.046160, -0.046670,
        -0.047160, -0.047600, -0.048000, -0.048350, -0.048640, -0.048900, -0.049100, -0.049250, -0.049340,
        -0.049380, -0.049360, -0.049270, -0.049140, -0.048940, -0.048670, -0.048350, -0.047970, -0.047520,
        -0.046990, -0.046410, -0.045760, -0.045050, -0.044270, -0.043420, -0.042500, -0.041520, -0.040470,
        -0.039350, -0.038160, -0.036920, -0.035610, -0.034240, -0.032800, -0.031320, -0.029790, -0.028210,
        -0.026600, -0.024940, -0.023270, -0.021560, -0.019840, -0.018110, -0.016390, -0.014660, -0.012960,
        -0.011270, -0.009600, -0.007970, -0.006380, -0.004840, -0.003350, -0.001910, -0.000550, 0.000760, 0.001980,
        0.003120, 0.004190, 0.005150, 0.006030, 0.006800, 0.007480, 0.008040, 0.008490, 0.008850, 0.009090,
        0.009220, 0.009250, 0.009180, 0.009010, 0.008750, 0.008400, 0.007970, 0.007470, 0.006900, 0.006280,
        0.005600, 0.004890, 0.004130, 0.003340, 0.002540, 0.001710, 0.000880, 0.000000)


def add_cleared_child(parent, child_name, attrib=None):
    # type: (etree._Element, str, Optional[dict]) -> etree._Element
    """Clears the child with the given name of the given parent with the given attributes and returns it.

    If a child doesn't exist yet with the given name and attributes, it is created.

    Parameters
    ----------
        parent : :obj:`etree._Element`
            Parent element.

        child_name : str
            Name of the child element.

        attrib : dict, optional
            Dictionary of attributes.

    Returns
    -------
        :obj:`etree._Element`
            The cleared child element.
    """
    for child in parent.findall(child_name):
        if attrib is None:
            child.clear()
            return child
        elif all([value == child.attrib[key] for (key, value) in attrib.items() if key in child.attrib]):
            child.clear()
            return child

    return etree.SubElement(parent, child_name, attrib)


def add_point(tree, parent, point_name, *args):
    # type: (etree._ElementTree, Union[str, etree._Element], str, *Union[float, Tuple[float]]) -> etree._Element
    """Safely creates a new point in the XML tree.

    Parameters
    ----------
        tree : :obj:`etree._ElementTree`
            Tree in which to add the point.

        parent : str or :obj:`etree._Element`
            XPath of the element or `etree._Element` under which to add the point.

        point_name : str
            Name of the point.

        *args
            Either three floats (x, y, z) or a length 3 tuple correspondingly, which describes the point.

    Returns
    -------
        :obj:`etree._Element`
            The `etree._Element` corresponding to the newly created point.
    """
    if len(args) == 1 and len(args[0]) == 3:
        x, y, z = args[0]
    elif len(args) == 3:
        x = args[0]
        y = args[1]
        z = args[2]
    else:
        raise ValueError('*args should be three floats or a tuple or length 3')

    if isinstance(parent, str):
        parent = xml_safe_create_element(tree, parent)

    point = add_cleared_child(parent, point_name)
    etree.SubElement(point, 'x').text = str(x)
    etree.SubElement(point, 'y').text = str(y)
    etree.SubElement(point, 'z').text = str(z)

    return point


def add_transform(tree, parent, scaling=(1, 1, 1), rotation=(0, 0, 0), translation=(0, 0, 0)):
    # type: (etree._ElementTree, Union[str, etree._Element], tuple, tuple, tuple) -> etree._Element
    """Safely creates a new CPACS transformation within the XML tree at the given XPath.

    Parameters
    ----------
        tree : :obj:`etree._ElementTree`
            Tree in which to add the transformation.

        parent : str or :obj:`etree._Element`
            XPath of an element or `etree._Element` under which to add the transformation.

        scaling, rotation, translation : tuple of float
            Tuples of length 3 corresponding to the x-, y-, and z-components of the scaling, rotation, and translation.

    Returns
    -------
        :obj:`etree._Element`
            `etree._Element` corresponding to the newly added transformation.
    """
    if len(scaling) != len(rotation) != len(translation) != 3:
        raise ValueError('scaling, rotation, and translation should be tuples of length 3')

    if isinstance(parent, str):
        parent = xml_safe_create_element(tree, parent)

    transform = add_cleared_child(parent, 'transformation')
    add_point(tree, transform, 'scaling', scaling)
    add_point(tree, transform, 'rotation', rotation)
    add_point(tree, transform, 'translation', translation)

    return transform


def add_spar_position(tree, parent, uid, element_uid, xsi):
    # type: (etree._ElementTree, Union[str, etree._Element], str, str, float) -> etree._Element
    """Safely creates a new CPACS spar position with the given XML tree.

    Parameters
    ----------
        tree : :obj:`etree._ElementTree`
            Tree in which to add the spar position.

        parent : str or :obj:`etree._Element`
            XPath of an element or `etree._Element` under which to add the spar position.

        uid, element_uid : str
            Unique identifiers of the spar position and the corresponding element.

        xsi : float
            Chordwise location of the spar position.

    Returns
    -------
        :obj:`etree._Element`
            `etree._Element` corresponding to the newly added spar position.
    """
    if isinstance(parent, str):
        parent = xml_safe_create_element(tree, parent)

    spar_pos = add_cleared_child(parent, 'sparPosition', {'uID': uid})
    etree.SubElement(spar_pos, 'name').text = uid
    etree.SubElement(spar_pos, 'elementUID').text = element_uid
    etree.SubElement(spar_pos, 'xsi').text = str(xsi)

    return spar_pos


def add_spar_segment(tree, parent, uid, start_pos_uid, end_pos_uid, mat_uid, t_web, t_top, t_bottom):
    # type: (etree._ElementTree, Union[str, etree._Element], str, str, str, str, float, float, float) -> etree._Element
    """Safely creates a new CPACS spar segment within the XML tree.

    Parameters
    ----------
        tree : :obj:`etree._ElementTree`
            Tree in which to add the spar segment.

        parent : str or :obj:`etree._Element`
            XPath of an element or `etree._Element` under which to add the spar segment.

        uid, start_pos_uid, end_pos_uid, mat_uid : str
            Unique identifiers of the spar segment, its start and end positions, and its material.

        t_web, t_top, t_bottom : float
            Thicknesses of the web, top, and bottom of the spar segment.

    Returns
    -------
        :obj:`etree._Element`
            `etree._Element` corresponding to the newly added spar segment.
    """
    if isinstance(parent, str):
        parent = xml_safe_create_element(tree, parent)

    spar_seg = add_cleared_child(parent, 'sparSegment', {'uID': uid})
    etree.SubElement(spar_seg, 'name').text = uid
    etree.SubElement(spar_seg, 'description').text = uid

    x_sparseg = tree.getpath(spar_seg)
    xml_safe_create_element(tree, '/'.join([x_sparseg, 'sparCrossSection/rotation']), 90)
    xml_safe_create_element(tree, '/'.join([x_sparseg, 'sparCrossSection/web1/relPos']), 0.5)
    xml_safe_create_element(tree, '/'.join([x_sparseg, 'sparCrossSection/web1/material/materialUID']), mat_uid)
    xml_safe_create_element(tree, '/'.join([x_sparseg, 'sparCrossSection/web1/material/thickness']), t_web)

    xml_safe_create_element(tree, '/'.join([x_sparseg, 'sparCrossSection/upperCap/area']), 0)
    xml_safe_create_element(tree, '/'.join([x_sparseg, 'sparCrossSection/upperCap/material/materialUID']), mat_uid)
    xml_safe_create_element(tree, '/'.join([x_sparseg, 'sparCrossSection/upperCap/material/thickness']), t_top)

    xml_safe_create_element(tree, '/'.join([x_sparseg, 'sparCrossSection/lowerCap/area']), 0)
    xml_safe_create_element(tree, '/'.join([x_sparseg, 'sparCrossSection/lowerCap/material/materialUID']), mat_uid)
    xml_safe_create_element(tree, '/'.join([x_sparseg, 'sparCrossSection/lowerCap/material/thickness']), t_bottom)

    uids = etree.SubElement(spar_seg, 'sparPositionUIDs')
    etree.SubElement(uids, 'sparPositionUID').text = start_pos_uid
    etree.SubElement(uids, 'sparPositionUID').text = end_pos_uid

    return spar_seg


def add_mass_description(tree, parent, element_name, mass, uid=None):
    # type: (etree._ElementTree, Union[str, etree._Element], str, float, Optional[str]) -> etree._Element
    """Safely adds a CPACS mass description to the given XML tree at the given parent.

    Parameters
    ----------
        tree : :obj:`etree._ElementTree`
            Tree in which to add the mass description.

        parent : str or :obj:`etree._Element`
            XPath of an element or `etree._Element` under which to add the mass description.

        element_name : str
            Name of the element.

        mass : float
            Value of the mass.

        uid : str, optional
            Unique identifier of the mass description.

    Returns
    -------
        :obj:`etree._Element`
            `etree._Element` corresponding to the newly added mass description.
    """
    if uid is None:
        uid = element_name

    if isinstance(parent, str):
        parent = xml_safe_create_element(tree, parent)

    elem = etree.SubElement(parent, element_name, {'uID': uid})
    etree.SubElement(elem, 'mass').text = str(mass)
    return elem


def add_mass(tree, parent, mass_name, mass, uid=None):
    # type: (etree._ElementTree, Union[str, etree._Element], str, float, Optional[str]) -> etree._Element
    """Safely creates a CPACS mass within the given XML tree at the given parent.

    Parameters
    ----------
        tree : :obj:`etree._ElementTree`
            Tree in which to add the mass.

        parent : str or :obj:`etree._Element`
            XPath of an element or `etree._Element` under which to add the mass.

        mass_name : str
            Name of the mass.

        uid : str, optional
            Unique identifier of the mass.

    Returns
    -------
        :obj:`etree._Element`
            `etree._Element` corresponding to the newly added mass.
    """
    if uid is None:
        uid = mass_name

    if isinstance(parent, etree._Element):
        parent = tree.getpath(parent)

    _mass = xml_safe_create_element(tree, '/'.join([parent, mass_name]))
    add_mass_description(tree, _mass, 'massDescription', mass, uid)
    return _mass


def add_cpacs_header(tree, name, creator, version, description, cpacs_version):
    # type: (etree._ElementTree, str, str, str, str, str) -> etree._Element
    """Add a CPACS header to the given XML tree root.

    Parameters
    ----------
        tree : :obj:`etree._ElementTree`
            Tree in which to add the header.

        name, creator, version, description, cpacs_version : str
            Name, creator, version identifier, and version of CPACS to put in the CPACS header information.

    Returns
    -------
        :obj:`etree._Element`
            `etree._Element` corresponding to the newly added header.
    """
    root = tree.getroot()
    header = etree.SubElement(root, 'header')
    etree.SubElement(header, 'name').text = name
    etree.SubElement(header, 'creator').text = creator
    etree.SubElement(header, 'version').text = version
    etree.SubElement(header, 'description').text = description
    etree.SubElement(header, 'cpacsVersion').text = cpacs_version
    return header


class WingObjectModel(AbstractDiscipline):
    """This discipline transforms a reduced aero-structural model of a wing into a CPACS file.

    The wing is modeled with a number of wing segments, n_wing_segments. Each wing segment has a span, b, sweep angle,
    Lambda, dihedral angle, Gamma, as well as four thicknesses, t_fs, t_rs, t_ts, and t_bs for the front spar,
    rear spar, top skin, and bottom skin associated with it. The sections joining two wing segments each have a chord
    length, c, thickness over chord ratio, t/c, twist angle, epsilon, from spar position, xsi_fs, and rear spar
    position, xsi_rs, associated with it. The most inboard section does not have a twist angle, but an incidence angle
    associated with it. Furthermore, a single thickness and material density are defined for the skin of the wing
    outside of the wingbox.

    Upon executing this discipline, a CPACS file containing the parameters of this reduced model is translated to a
    regular CPACS file. The result should be a wing that has the same geometry as was described by the reduced model.
    """

    def __init__(self, n_wing_segments=2):
        super(WingObjectModel, self).__init__()
        self.n_wing_segments = n_wing_segments

    @property
    def creator(self):
        return 'D. de Vries'

    @property
    def description(self):
        return 'Wing object model'

    def generate_input_xml(self):
        # type: () -> str
        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        xml_safe_create_element(doc, x_c, np.zeros(self.n_wing_segments + 1))
        xml_safe_create_element(doc, x_tc, np.zeros(self.n_wing_segments + 1))
        xml_safe_create_element(doc, x_epsilon, np.zeros(self.n_wing_segments))
        xml_safe_create_element(doc, x_b, np.zeros(self.n_wing_segments))
        xml_safe_create_element(doc, x_Lambda, np.zeros(self.n_wing_segments))
        xml_safe_create_element(doc, x_Gamma, np.zeros(self.n_wing_segments))
        xml_safe_create_element(doc, x_xsi_fs, np.zeros(self.n_wing_segments + 1))
        xml_safe_create_element(doc, x_xsi_rs, np.zeros(self.n_wing_segments + 1))
        xml_safe_create_element(doc, x_t_fs, np.zeros(self.n_wing_segments))
        xml_safe_create_element(doc, x_t_rs, np.zeros(self.n_wing_segments))
        xml_safe_create_element(doc, x_t_ts, np.zeros(self.n_wing_segments))
        xml_safe_create_element(doc, x_t_bs, np.zeros(self.n_wing_segments))
        xml_safe_create_element(doc, x_incidence, 0.)
        xml_safe_create_element(doc, x_t_skin, 0.)
        xml_safe_create_element(doc, x_rho_skin, 0.)

        xml_safe_create_element(doc, x_m_fixed, 0.)
        xml_safe_create_element(doc, x_m_payload, 0.)
        xml_safe_create_element(doc, x_m_wing_copy, 0.)
        xml_safe_create_element(doc, x_m_fuel_copy, 0.)
        xml_safe_create_element(doc, x_m_mlw, 0.)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def generate_output_xml(self):
        # type: () -> str
        _1 = np.zeros(self.n_wing_segments + 1)
        _2 = np.zeros(self.n_wing_segments)
        _3 = np.zeros((3, self.n_wing_segments + 1))

        return WingObjectModel.write_data(0, 0, _1, _1, _1, _3, _1, _1, _2, _2, _2, _2, _2, 0, 0, 0, 0, 0, 0, 0, _2)

    @staticmethod
    def write_data(*args):
        # type: (*(str, str, Sized)) -> str
        """Utility method that writes all output variables to the given XML file.

        Parameters
        ----------
            *args
                (S_ref, c_ref, c, t/c, twists, x_LE,
                xsi_fs, xsi_rs, t_fs, t_rs, t_ts, t_bs,
                m_skin, m_fuel, m_wing, m_fixed, m_payload, m_mlw, f_m_sys, f_m_wings, m_wingbox)

        Returns
        -------
            str
                String representing the output XML file.
        """
        s_ref, c_ref, c, tc, twists, x_le, xsi_fs, xsi_rs, t_fs, t_rs, t_ts, t_bs, m_skin, m_fuel, m_wing, m_fixed, m_payload, m_mlw, f_m_sys, f_m_wings, m_wingbox = args

        root = etree.Element('cpacs')
        doc = etree.ElementTree(root)

        add_cpacs_header(doc, 'WingObjectModel', 'Automatically generated from python', '1.0', 'Wing object model',
                         '2.3')

        xml_safe_create_element(doc, '/'.join([x_model, 'name']), 'Wing Definition')

        xml_safe_create_element(doc, x_ref_area, s_ref)
        xml_safe_create_element(doc, x_ref_length, c_ref)
        add_point(doc, x_ref, 'point', 0, 0, 0)

        xml_safe_create_element(doc, x_wing)
        add_transform(doc, x_wing)

        for i in range(len(c)):
            _x_sec = x_sec % i
            elem_sec = xml_safe_create_element(doc, _x_sec)
            etree.SubElement(elem_sec, 'name').text = 'Section %d' % i
            add_transform(doc, elem_sec, (c[i], 1, tc[i] * c[i]), (0, np.rad2deg(twists[i]), 0), tuple(x_le[:, i]))

            elem_elems = etree.SubElement(elem_sec, 'elements')
            elem_elem = etree.SubElement(elem_elems, 'element', {'uID': 'elem_%d' % i})
            etree.SubElement(elem_elem, 'name').text = 'Element %d' % i
            etree.SubElement(elem_elem, 'airfoilUID').text = 'af'
            add_transform(doc, elem_elem)

            x_pos = '/'.join([x_wing, r"positionings/positioning"])
            if i != 0:
                x_pos += '[%d]' % (i + 1)
            pos = xml_safe_create_element(doc, x_pos)
            etree.SubElement(pos, 'length').text = str(0)
            etree.SubElement(pos, 'sweepAngle').text = str(0)
            etree.SubElement(pos, 'dihedralAngle').text = str(0)
            if i != 0:
                etree.SubElement(pos, 'fromSectionUID').text = 'sec_%d' % (i - 1)
            etree.SubElement(pos, 'toSectionUID').text = 'sec_%d' % i

        for i in range(len(c) - 1):
            x_seg = '/'.join([x_wing, r"segments/segment[@uID='seg_%d']" % i])
            seg = xml_safe_create_element(doc, x_seg)
            etree.SubElement(seg, 'name').text = 'Segment %d' % i
            etree.SubElement(seg, 'fromElementUID').text = 'elem_%d' % i
            etree.SubElement(seg, 'toElementUID').text = 'elem_%d' % (i + 1)

            compseg = xml_safe_create_element(doc, x_compseg % i)
            etree.SubElement(compseg, 'name').text = 'ComponentSegment %d' % i
            etree.SubElement(compseg, 'fromElementUID').text = 'elem_%d' % i
            etree.SubElement(compseg, 'toElementUID').text = 'elem_%d' % (i + 1)

            add_spar_position(doc, x_sparposs % i, 'fs_%d_r' % i, 'elem_%d' % i, xsi_fs[i])
            add_spar_position(doc, x_sparposs % i, 'fs_%d_t' % i, 'elem_%d' % (i + 1), xsi_fs[i + 1])
            add_spar_position(doc, x_sparposs % i, 'rs_%d_r' % i, 'elem_%d' % i, xsi_rs[i])
            add_spar_position(doc, x_sparposs % i, 'rs_%d_t' % i, 'elem_%d' % (i + 1), xsi_rs[i + 1])

            add_spar_segment(doc, x_sparsegs % i, 'fs_%d' % i, 'fs_%d_r' % i, 'fs_%d_t' % i,
                             'mat_al', t_fs[i], t_ts[i], t_bs[i])
            add_spar_segment(doc, x_sparsegs % i, 'rs_%d' % i, 'rs_%d_r' % i, 'rs_%d_t' % i,
                             'mat_al', t_rs[i], t_ts[i], t_bs[i])

        # x_mbd = '/'.join([x_model, 'analyses/massBreakdown'])

        m_empty = m_wing + m_fixed
        m_sys = m_empty * f_m_sys
        m_struct = m_empty - m_sys
        m_wings = m_struct * f_m_wings
        m_wing = m_wing * (1. - f_m_sys)

        m_mzf = m_empty + m_payload
        m_mto = m_mzf + m_fuel
        m_mrm = m_mto * 1.01

        add_mass(doc, x_mbd, 'fuel', m_fuel, 'mFuel')
        add_mass(doc, x_mbd, 'payload', m_payload, 'mPayload')
        m_oem = add_mass(doc, x_mbd, 'mOEM', m_empty)
        m_em = add_mass(doc, m_oem, 'mEM', m_empty)
        add_mass(doc, m_em, 'mSystems', m_sys)

        m_structure = add_mass(doc, m_em, 'mStructure', m_struct)
        m_wings = add_mass(doc, m_structure, 'mWingsStructure', m_wings)
        m_wing = add_mass(doc, m_wings, 'mWingStructure', m_wing)

        for i in range(0, len(c) - 1):
            x_m_compseg = '/'.join([doc.getpath(m_wing), 'mComponentSegment[%d]' % (i + 1)])
            m_compseg = xml_safe_create_element(doc, x_m_compseg)
            add_mass_description(doc, m_compseg, 'massDescription', m_skin[i] + m_wingbox[i], 'mWing_%d' % i)
            add_mass(doc, m_compseg, 'mWingBox', m_wingbox[i], 'mWingbox_%d' % i)
            add_mass(doc, x_mSkins % (i + 1), 'mSkins', m_skin[i], 'mSkins_%d' % i)

        m_des = etree.SubElement(m_oem.getparent(), 'designMasses')
        add_mass_description(doc, m_des, 'mMLM', m_mlw)
        add_mass_description(doc, m_des, 'mMRM', m_mrm)
        add_mass_description(doc, m_des, 'mTOM', m_mto)
        add_mass_description(doc, m_des, 'mZFM', m_mzf)

        x_af = r"/cpacs/vehicles/profiles/wingAirfoils/wingAirfoil[@uID='af']"
        af = xml_safe_create_element(doc, x_af)
        etree.SubElement(af, 'name').text = 'Airfoil'
        point_list = etree.SubElement(af, 'pointList')

        etree.SubElement(point_list, 'x', {'mapType': 'vector'}).text = ';'.join([str(_) for _ in af_x])
        etree.SubElement(point_list, 'y', {'mapType': 'vector'}).text = ';'.join([str(_) for _ in len(af_x) * [0.0]])
        etree.SubElement(point_list, 'z', {'mapType': 'vector'}).text = ';'.join([str(_) for _ in af_z])

        mat = xml_safe_create_element(doc, '/'.join([x_vehicles, r"materials/material[@uID='mat_al']"]))
        etree.SubElement(mat, 'name').text = 'Al7075A'
        etree.SubElement(mat, 'rho').text = str(2180)
        etree.SubElement(mat, 'k11').text = str(71.7E9)
        etree.SubElement(mat, 'k12').text = str(26.9E9)
        etree.SubElement(mat, 'sig11').text = str(572E6)
        etree.SubElement(mat, 'sig12').text = str(331E6)

        return etree.tostring(doc, encoding='utf-8', pretty_print=True, xml_declaration=True)

    @staticmethod
    def execute(in_file, out_file='WOM-output-loc.xml'):
        """Translate the reduced model parameters to CPACS format."""
        tree = etree.parse(in_file)

        c = np.array(tree.xpath(x_c)[0].text.split(';'), dtype=float)
        tc = np.array(tree.xpath(x_tc)[0].text.split(';'), dtype=float)
        epsilon = np.array(tree.xpath(x_epsilon)[0].text.split(';'), dtype=float)
        b = np.array(tree.xpath(x_b)[0].text.split(';'), dtype=float)
        sweep = np.array(tree.xpath(x_Lambda)[0].text.split(';'), dtype=float)
        dihed = np.array(tree.xpath(x_Gamma)[0].text.split(';'), dtype=float)
        xsi_fs = np.array(tree.xpath(x_xsi_fs)[0].text.split(';'), dtype=float)
        xsi_rs = np.array(tree.xpath(x_xsi_rs)[0].text.split(';'), dtype=float)
        t_fs = np.array(tree.xpath(x_t_fs)[0].text.split(';'), dtype=float)
        t_rs = np.array(tree.xpath(x_t_rs)[0].text.split(';'), dtype=float)
        t_ts = np.array(tree.xpath(x_t_ts)[0].text.split(';'), dtype=float)
        t_bs = np.array(tree.xpath(x_t_bs)[0].text.split(';'), dtype=float)
        incidence = np.array(tree.xpath(x_incidence)[0].text.split(';'), dtype=float)
        t_skin = np.array(tree.xpath(x_t_skin)[0].text.split(';'), dtype=float)
        rho_skin = np.array(tree.xpath(x_rho_skin)[0].text.split(';'), dtype=float)

        m_fuel = float(tree.xpath(x_m_fuel_copy)[0].text)
        m_wing = float(tree.xpath(x_m_wing_copy)[0].text)

        if m_fuel == 0.:
            m_fuel = float(tree.xpath(x_m_fuel_init)[0].text)
        if m_wing == 0.:
            m_wing = float(tree.xpath(x_m_wing_init)[0].text)

        m_fixed = float(tree.xpath(x_m_fixed)[0].text)
        m_payload = float(tree.xpath(x_m_payload)[0].text)
        m_mlw = float(tree.xpath(x_m_mlw)[0].text)
        f_m_sys = float(tree.xpath(x_f_m_sys)[0].text)
        f_m_wings = float(tree.xpath(x_f_m_wings)[0].text)

        n_wing_segments = len(b)

        s_ref = sum(b * (c[:-1] + c[1:]))

        c_ref = c[:]
        for i in range(0, n_wing_segments):
            c_ref = 2. / 3. * (c_ref[:-1] ** 2 + c_ref[:-1] * c_ref[1:] + c_ref[1:] ** 2) / (c_ref[1:] + c_ref[:-1])
        c_ref = c_ref[0]

        dx_c4 = np.zeros((3, n_wing_segments))
        dx_c4[1, :] = b[:]

        x_c4 = np.zeros((3, n_wing_segments + 1))
        c_sweep, s_sweep = np.cos(sweep), np.sin(sweep)
        c_dihed, s_dihed = np.cos(dihed), np.sin(dihed)

        for i in range(n_wing_segments):
            rot_sweep = np.matrix([(c_sweep[i], s_sweep[i], 0), (-s_sweep[i], c_sweep[i], 0), (0, 0, 1)])
            rot_dihed = np.matrix([(1, 0, 0), (0, c_dihed[i], -s_dihed[i]), (0, s_dihed[i], c_dihed[i])])
            x_c4[:, i + 1] = np.matmul(rot_dihed * rot_sweep, dx_c4[:, i]) + x_c4[:, i]

        dx_le = np.zeros((3, n_wing_segments + 1))
        dx_le[0, :] = -.25 * c

        x_le = np.zeros((3, n_wing_segments + 1))
        twists = np.concatenate(([0.], epsilon)) + incidence
        c_twist, s_twist = np.cos(twists), np.sin(twists)
        for i in range(n_wing_segments + 1):
            rot_twist = np.matrix([(c_twist[i], 0, s_twist[i]), (0, 1, 0), (-s_twist[i], 0, c_twist[i])])
            x_le[:, i] = np.matmul(rot_twist, dx_le[:, i]) + x_c4[:, i]

        length_out = c * (1. - xsi_rs + xsi_fs)
        area_out = 0.5 * (length_out[:-1] + length_out[1:]) * b
        m_skin = 2. * area_out * t_skin * rho_skin

        m_wingbox = m_wing * area_out / sum(area_out)

        xml = WingObjectModel.write_data(s_ref, c_ref,
                                         c, tc, twists,
                                         x_le, xsi_fs, xsi_rs,
                                         t_fs, t_rs, t_ts,
                                         t_bs, m_skin,
                                         m_fuel, m_wing, m_fixed, m_payload, m_mlw, f_m_sys, f_m_wings, m_wingbox)
        with open(out_file, 'w') as f:
            f.write(xml)
