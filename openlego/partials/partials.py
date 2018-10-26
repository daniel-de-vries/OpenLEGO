#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2018 D. de Vries

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains the definition of the `Partials` object.
"""
from __future__ import absolute_import, division, print_function

import os
import warnings

from lxml import etree
from typing import Union, List, Optional, Any

from openlego.utils.general_utils import parse_string
from openlego.utils.xml_utils import value_to_xml


dir_path = os.path.dirname(os.path.abspath(__file__))
xsd_file_path = os.path.join(dir_path, 'partials.xsd')
xsi_schema_location = 'file:///' + xsd_file_path

schema = etree.XMLSchema(file=xsi_schema_location)
parser = etree.XMLParser(schema=schema)


class Partials(object):

    def __init__(self, file=None):
        # type: (Optional[str]) -> None
        """Initialize `Partials` object.

        Parameters
        ----------
            file : str, optional
                Path to the partials XML file to initialize from.
        """
        super(Partials, self).__init__()

        if file is None:
            self._tree = etree.ElementTree(etree.Element('partials'), parser=parser)    # type: etree._ElementTree
        else:
            self._tree = etree.parse(file, parser)

    @property
    def _elem_root(self):
        # type: () -> etree._Element
        """Root `_Element` of the partials XML file."""
        return self._tree.getroot()

    def get_partials(self, of=None):
        # type: (Optional[str]) -> dict
        """Get a dictionary with the partials stored in the XML file.

        Parameters
        ----------
            of : str, optional
                Name of the parameter to get the partials of

        Returns
        -------
            dict
                In the form partials['from_param_name']['wrt_param_name'] = sensitivity_value if no `of` is given, or
                in the form partials['wrt_param_name'] = sensitivity_value if `of` is given.
        """
        partials = dict()

        if of is not None:
            elem_of = self._tree.xpath('/partials/of[@uid="{}"]'.format(of))
            if len(elem_of):
                for elem_wrt in elem_of:
                    wrt_uid = elem_wrt.get('uid')
                    value = parse_string(elem_wrt.text)
                    partials.update({wrt_uid: value})
        else:
            for elem_of in self._elem_root:
                of_uid = elem_of[0].text

                if of_uid not in partials:
                    partials.update({of_uid: dict()})

                for elem_wrt in elem_of[1:]:
                    wrt_uid = elem_wrt[0].text

                    if len(elem_wrt) == 2:
                        value = parse_string(elem_wrt[1].text)
                    else:
                        value = 0.
                    partials[of_uid].update({wrt_uid: value})

        return partials

    def declare_partials(self, of, wrt, val=None):
        # type: (str, Union[str, List[str]], Optional[Any]) -> None
        """Declare a set of partials that is provided.

        Parameters
        ----------
            of : str
                Name of the parameter of which the derivative is to be taken.

            wrt : str or Iterable[str]
                Name(s) of parameters w.r.t. which the derivatives are to be taken.

            val : any, optional
                Optional value(s) of partials.

        Notes
        -----
            If `val` is given and `wrt` is a list, `val` should have the same length as `wrt`.
        """
        if not isinstance(wrt, list):
            wrt = [wrt]
            if val is not None:
                val = [val]
        else:
            wrt = set(wrt)

        elem_root = self._elem_root

        x_of = "/partials/of[uid='{}']".format(of)
        elem_of = self._tree.xpath(x_of)

        if not len(elem_of):
            elem_of = etree.SubElement(elem_root, 'of')
            etree.SubElement(elem_of, 'uid').text = of
        else:
            elem_of = elem_of[0]

        for i, _wrt in enumerate(wrt):
            x_wrt = x_of + "/wrt[uid='{}']".format(_wrt)
            elem_wrt = self._tree.xpath(x_wrt)

            if not len(elem_wrt):
                elem_wrt = etree.SubElement(elem_of, 'wrt')
                etree.SubElement(elem_wrt, 'uid').text = _wrt
                if val is not None:
                    value_to_xml(etree.SubElement(elem_wrt, 'value'), val[i])
            else:
                warnings.warn(
                    'Partial from {} to {} is defined more than once. Last occurrence take precedence.'
                        .format(of, _wrt))
                value_to_xml(elem_wrt[0][1], val[i])

    def add_partials(self, partials):
        # type: (dict) -> None
        """Add a set of partials to the XML file.

        Parameters
        ----------
            partials : dict
                Dictionary of the partials.
        """
        for param_uid, param in partials.items():
            self.declare_partials(param_uid, param.keys(), param.values())

    def write(self, file):
        # type: (str) -> None
        """Write the current state of the class to a partials XML file.

        Parameters
        ----------
            file : str
                Path of the file to write to.
        """
        if not schema.validate(self._tree):
            raise RuntimeError('Something is wrong.. XML is not a valid partials file.')

        self._tree.write(file, encoding='utf-8', pretty_print=True, xml_declaration=True)

    def get_string(self):
        # type: () -> str
        """Return the current state of the class as a partials XML string.

        Returns
        -------
            str
                String representation of a partials XML file.
        """
        if not schema.validate(self._tree):
            raise RuntimeError('Something is wrong.. XML is not a valid partials file.')

        return etree.tostring(self._tree, encoding='utf-8', pretty_print=True, xml_declaration=True)