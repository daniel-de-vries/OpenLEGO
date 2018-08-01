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

This file contains a set of CMDOWS utility functions.
"""
from __future__ import absolute_import, division, print_function

from copy import copy
from lxml.etree import _Element
from typing import Tuple, Union


def get_related_parameter_uid(elem_param, full_xml):
    # type: (_Element, _Element) -> Tuple[str, str]
    """Function to retrieve the UID of the related parameter. This UID refers to the original local in the XML file of
     the parameter and is used when a parameter is found in a CMDOWS file which is an architecture type or a higher
     instance.

    Parameters
    ----------
        elem_param : _Element
            Element of the parameter to be mapped.
        full_xml : _Element
            Element of the full XML file in which the related parameter UID should be found.

    Returns
    -------
        param : str
            Original parameter UID for which the related UID is searched for.
        mapped : str
            The found related parameter UID.
    """
    param = elem_param.attrib['uID']
    if isinstance(elem_param.find('relatedParameterUID'), _Element):
        mapped = elem_param.find('relatedParameterUID').text
    elif isinstance(elem_param.find('relatedInstanceUID'), _Element):
        related_instance = get_element_by_uid(full_xml, elem_param.find('relatedInstanceUID').text)
        mapped = related_instance.find('relatedParameterUID').text
    else:
        raise AssertionError('Could not map element {}.'.format(param))
    return str(param), str(mapped)


def get_element_by_uid(xml, uid):
    # type: (_Element, str) -> _Element
    """Method to get the element based on a UID value.

    Parameters
    ----------
        xml : _Element
            Element of an XML file where the uID should be found.

        uid : str
            uID to be found.

    Returns
    -------
        _Element
            Element with the right UID.
    """
    xpath_expression = get_uid_search_xpath(uid)
    els = xml.xpath(xpath_expression)
    if len(els) > 1:
        raise AssertionError('Multiple elements with UID ' + uid + ' found. Use "check_uids()" to check if all UIDs'
                                                                   ' are unique.')
    elif len(els) == 0:
        raise AssertionError('Could not find element with UID ' + uid + '.')
    return els[0]


def get_uid_search_xpath(uid):
    # type: (str) -> str
    """Method to get the XPath expression for a UID that might contain quote characters.

    Parameters
    ----------
        uid : str
            Original UID string with XPath expression.

    Returns
    -------
        str
            Processed XPath expression to escape quote characters using "concat()".
    """
    if '"' in uid or '&quot;' in uid:
        uid_concat = "concat('%s')" % uid.replace('&quot;', "\',\'\"\',\'").replace('"', "\',\'\"\',\'")
        return './/*[@uID=' + uid_concat + ']'
    else:
        return './/*[@uID="' + uid + '"]'


def get_loop_nesting_obj(elem):
    # type: (_Element) -> Union[list, dict]
    """Function to make an object of the loop hierarchy based on the loopNesting element in a CMDOWS file.

    Parameters
    ----------
        elem : _Element
            Element in the XML file (loopNesting or one of its subelements)

    Returns
    -------
        list or dict
            A list object containing dictionaries and string entries to represent the loopNesting element.

    Raises
    ------
        AssertionError
            If the provided element is not formatted correctly.
    """
    basic_list = []
    basic_dict = {}
    d = copy(elem.attrib)
    if elem.tag == 'loopNesting' or elem.tag == 'loopElements' or elem.tag == 'functionElements':
        for x in elem.iterchildren():
            basic_list.append(get_loop_nesting_obj(x))
        if len(basic_list) == 1 and isinstance(basic_list[0], list):
            return basic_list[0]
        else:
            return basic_list
    elif elem.tag == 'loopElement':
        basic_dict[d['relatedUID']] = []
        for x in elem.iterchildren():
            basic_dict[d['relatedUID']].extend(get_loop_nesting_obj(x))
        return basic_dict
    elif elem.tag == 'functionElement':
        return elem.text
    else:
        raise AssertionError('Something went wrong in this function...')
