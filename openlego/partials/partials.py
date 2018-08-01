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

    def get_partials(self, src=None):
        # type: (Optional[src]) -> dict
        """Get a dictionary with the partials stored in the XML file.

        Parameters
        ----------
            src : str, optional
                Name of the source parameter to get the partials from

        Returns
        -------
            dict
                In the form partials['from_param_name']['to_param_name'] = sensitivity_value if no `src` is given, or
                in the form partials['to_param_name'] = sensitivity_value if `src` is given.
        """
        partials = dict()

        if src is not None:
            elem_param = self._tree.xpath('/partials/parameter[uid="{}"]'.format(src))
            if len(elem_param):
                for elem_partial in elem_param[1]:
                    param = elem_partial[0].text
                    value = parse_string(elem_partial[1].text)
                    partials.update({param: value})
        else:
            for elem_param in self._elem_root:
                uid = elem_param[0].text

                if uid not in partials:
                    partials.update({uid: dict()})

                for elem_partial in elem_param[1]:
                    param = elem_partial[0].text
                    if len(elem_partial) > 1:
                        value = parse_string(elem_partial[1].text)
                    else:
                        value = 0.
                    partials[uid].update({param: value})

        return partials

    def declare_partials(self, src, tgt, val=None):
        # type: (str, Union[str, List[str]], Optional[Any]) -> None
        """Declare a set of partials that is provided.

        Parameters
        ----------
            src : str
                Name of the source parameter.

            tgt : str or Iterable[str]
                Name(s) of target parameters.

            val : any, optional
                Optional value(s) of partials.

        Notes
        -----
            If `val` is given and `tgt` is a list, `val` should have the same length as `tgt`.
        """
        if not isinstance(tgt, list):
            tgt = [tgt]
            if val is not None:
                val = [val]

        elem_root = self._elem_root

        x_param = "/partials/parameter[uid='{}']".format(src)
        elem_param = self._tree.xpath(x_param)

        if not len(elem_param):
            elem_param = etree.SubElement(elem_root, 'parameter')
            elem_param_uid = etree.SubElement(elem_param, 'uid')
            elem_param_uid.text = src

            elem_partials = etree.SubElement(elem_param, 'partials')
        else:
            elem_partials = elem_param[0][1]

        for i, t in enumerate(tgt):
            x_partial = '/'.join([x_param, "partials/partial[uid='{}']"]).format(t)
            elem_partial = self._tree.xpath(x_partial)

            if not len(elem_partial):
                elem_partial = etree.SubElement(elem_partials, 'partial')
                elem_param_uid = etree.SubElement(elem_partial, 'uid')
                elem_param_uid.text = t
            else:
                warnings.warn(
                    'Partial from {} to {} is defined more than once. Last occurrence take precedence.'
                    .format(src, t))

            if val is not None:
                elem_value = etree.SubElement(elem_partial, 'value')
                value_to_xml(elem_value, val[i])

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