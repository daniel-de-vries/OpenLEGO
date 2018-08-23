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

import warnings
from copy import copy
from lxml.etree import _Element
from typing import Tuple, Union, Any, List

from openlego.utils.general_utils import change_object_type


def get_related_parameter_uid(elem_or_uid_param, full_xml):
    # type: (_Element, _Element) -> Tuple[str, str]
    """Function to retrieve the UID of the related parameter. This UID refers to the original local in the XML file of
     the parameter and is used when a parameter is found in a CMDOWS file which is an architecture type or a higher
     instance.

    Parameters
    ----------
        elem_or_uid_param : _Element or str
            Element of the parameter to be mapped or string with UID.
        full_xml : _Element
            Element of the full XML file in which the related parameter UID should be found.

    Returns
    -------
        param : str
            Original parameter UID for which the related UID is searched for.
        mapped : str
            The found related parameter UID.
    """
    if isinstance(elem_or_uid_param, _Element):
        param = elem_or_uid_param.attrib['uID']
        elem_param = elem_or_uid_param
    elif isinstance(elem_or_uid_param, str):
        param = elem_or_uid_param
        elem_param = get_element_by_uid(full_xml, param)
    else:
        raise IOError('elem_or_uid_param input is not of the right type.')
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


def get_opt_setting_safe(opt_elem, setting, default, expected_type='str'):
    # type: (_Element, str, Any, str) -> Union[str, int, float]
    """Function to read out an optimizer setting from a CMDOWS file, and to provide a default value (and warning) if the
    setting is not found.

    Parameters
    ----------
        opt_elem : _Element
            The lxml element of the optimizer block in the CMDOWS file.

        setting : str
            The setting to be found.

        default : Any
            The default value of the setting if it is not found in the element.

        expected_type : str
            The expected type of the setting (str, int, float) so that the XML string value can be changed accordingly.

    Returns
    -------
        Union[str, int, float]
            The optimizer setting that was found or its default value if it was not found.
    """
    if isinstance(opt_elem.find('settings/{}'.format(setting)), _Element):
        opt_setting = opt_elem.find('settings/{}'.format(setting)).text
    else:
        warnings.warn('Setting "{}" not specified for optimizer element "{}", setting to default "{}".'
                      .format(setting, opt_elem.attrib['uID'], default))
        opt_setting = default
    return change_object_type(opt_setting, expected_type)


def get_doe_setting_safe(doe_elem, setting, default, expected_type='str', doe_method=None, required_for_doe_methods=None):
    # type: (_Element, str, Any, str, str, List[str]) -> Union[str, int, float]
    """Function to read out a DOE setting from a CMDOWS file based on whether that setting is required, and to provide
    a default value (and warning) if the setting is not found.

    Parameters
    ----------
        doe_elem : _Element
            The lxml element of the DOE block in the CMDOWS file.

        setting : str
            The setting to be found.

        default : Any
            The default value of the setting if it is not found in the element.

        expected_type : str
            The expected type of the setting (str, int, float) so that the XML string value can be changed accordingly.

        doe_method : str
            The DOE method (LHS, Box-Behnken, etc.) of the DOE block.

        required_for_doe_methods : List[str]
            The DOE methods for which this setting must be found.

    Returns
    -------
        doe_setting : Union[str, int, float]
            The DOE setting that was found or its default value if it was not found.
    """
    if isinstance(doe_elem.find('settings/{}'.format(setting)), _Element):
        doe_setting = doe_elem.find('settings/{}'.format(setting)).text
    else:
        if required_for_doe_methods:
            if doe_method in required_for_doe_methods:
                warnings.warn('Setting "{}" not specified for DOE element "{}", setting to default "{}".'
                              .format(setting, doe_elem.attrib['uID'], default))
                doe_setting = default
            else:
                doe_setting = None
        else:
            doe_setting = None
    if doe_setting:
        return change_object_type(doe_setting, expected_type)
    else:
        return doe_setting


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
