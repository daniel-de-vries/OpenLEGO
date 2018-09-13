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

This file contains a set of XML utility functions.
"""
from __future__ import absolute_import, division, print_function

import re
from collections import OrderedDict
from shutil import copyfile

import numpy as np
from lxml import etree
from typing import Optional, Union, List

# Patterns for XML attribute names and values
pttrn_attr_val = r'([-.0-9:A-Z_a-z]+?)'
pttrn_attr_name = r'([:A-Z_a-z][0-9:A-Z_a-z]*?)'

# Expressions used to replace illegal characters in an XPath to legal characters within an OpenMDAO variable name.
# They have to be executed in this order when going from XPaths to variables, and in reversed order the other way.
# If they are not executed in this order the transformation may not be reversible.
repl_atr = r':_:\1:_:\2'       # An attribute ([@name="value"]) becomes :_:name:_:value
repl_ind = r':_:_\1'           # An index ([2]) becomes :_:_2
repl_dot = ':__:'              # A dot (.) becomes :__:
repl_str = ':___:'             # A star (*) becomes :___:
repl_qtm = ':____:'            # A question mark (?) becomes :____:
repl_emm = ':_____:'           # An exclamation mark (!) becomes :_____:

repl_atr_inv = r'[@\1="\2"]'
repl_ind_inv = r'[\1]'
repl_dot_inv = '.'
repl_str_inv = '*'
repl_qtm_inv = '?'
repl_emm_inv = '!'

# Regular expressions to match attributes and indices within valid XPaths
re_atr = re.compile(r'\[@' + pttrn_attr_name + "=['\"]" + pttrn_attr_val + "['\"]\]")
re_ind = re.compile(r'\[([0-9]+?)\]')

# Regular expressions to match attributes and indices within OpenMDAO variables transformed from xpaths
re_atr_inv = re.compile(r':_:' + pttrn_attr_val + ':_:' + pttrn_attr_val + r'(?=/|$)')
re_ind_inv = re.compile(r':_:_([0-9]+?)(?=/|$)')

parser = etree.XMLParser(remove_blank_text=True, encoding='utf-8')
find_text = etree.XPath('//text()')


def xpath_to_param(xpath):
    # type: (str) -> str
    """Convert an XML XPath to a valid ``OpenMDAO`` parameter name.

    Parameters
    ----------
        xpath : str
            XPath to convert.

    Returns
    -------
        str
            Valid ``OpenMDAO`` parameter name.
    """
    param = re_atr.sub(repl_atr, xpath)
    param = re_ind.sub(repl_ind, param)
    param = param.replace(repl_dot_inv, repl_dot)
    param = param.replace(repl_str_inv, repl_str)
    param = param.replace(repl_qtm_inv, repl_qtm)
    param = param.replace(repl_emm_inv, repl_emm)
    return param


def param_to_xpath(param):
    # type: (str) -> str
    """Convert an ``OpenMDAO`` parameter name to the corresponding XML XPath.

    This function is the inverse of `xpath_to_param()`.

    Parameters
    ----------
        param : str
            Valid ``OpenMDAO`` parameter name.

    Returns
    -------
        str
            Corresponding XML XPath.
    """
    xpath = param.replace(repl_emm, repl_emm_inv)
    xpath = xpath.replace(repl_qtm, repl_qtm_inv)
    xpath = xpath.replace(repl_str, repl_str_inv)
    xpath = xpath.replace(repl_dot, repl_dot_inv)
    xpath = re_ind_inv.sub(repl_ind_inv, xpath)
    xpath = re_atr_inv.sub(repl_atr_inv, xpath)
    return xpath


def value_to_xml(elem, value):
    if isinstance(value, np.ndarray):
        value = np.atleast_1d(value).flatten()

    if isinstance(value, np.ndarray):
        if value.size == 1:
            elem.text = str('{:.16f}'.format(value[0]))
        else:
            elem.text = ';'.join([str('{:.16f}'.format(v)) for v in value[:]])
            elem.attrib.update({'mapType': 'vector'})
    elif isinstance(value, float):
        elem.text = str('{:.16f}'.format(value))
    else:
        elem.text = str(value)


def xml_to_dict(xml):
    # type: (Union[str, etree._ElementTree]) -> OrderedDict
    """Convert an XML file to a python dictionary with all valued elements as values with their full XPaths as keys.

    Parameters
    ----------
        xml : str or :obj:`etree._ElementTree`
            Path to or `etree._ElementTree` of an XML file.

    Returns
    -------
        :obj:`OrderedDict`
            `OrderedDict` representing the XML file in file order.
    """
    if isinstance(xml, str):
        xml = etree.parse(xml, parser)

    _dict = OrderedDict()
    for text in find_text(xml):
        # Construct 'augmented' XPath for this element, including attributes
        xpath = ''
        child = text.getparent()
        while child is not None:
            parent = child.getparent()

            tag = child.tag
            for name, value in child.items():
                # Exclude special purpose attribute: mapType
                if name != 'mapType':
                    tag += r'[@%s="%s"]' % (name, value)

            if parent is not None:
                siblings = parent.findall(tag)
                if len(siblings) > 1:
                    tag += '[%d]' % (siblings.index(child) + 1)

            xpath = '/'.join([tag, xpath])
            child = parent

        # Try to convert the text into a float or a list of floats
        try:
            value = float(text)
        except ValueError:
            try:
                value = np.array(text.split(';'), dtype=float)
            except ValueError:
                value = str(text)

        # Update the dict with this element
        _dict.update({'/' + xpath[:-1]: value})

    return _dict


def xml_safe_create_element(
        tree,       # type: etree._ElementTree
        xpath,      # type: str
        value=None  # type: Optional[Union[str, int, float, List[Union[str, int, float]], np.ndarray]]
):
    # type: (...) -> etree._Element
    """Create an element at the given XML XPath with the given value.

    This method ensures that all elements implied by the given X-Path exist.

    Supplying a value is optional. If no value is supplied an empty XML node is created at the deepest level implied by
    the XPath.

    Parameters
    ----------
        tree : :obj:`etree._ElementTree`
            `etree._ElementTree` in which to create the element.

        xpath : str
            XPath to ensure.

        value : str or int or float or list of str or list of int or list of float or :obj:`np.ndarray`
            Optional value to write at the deepest node of the ensured XPath.

    Returns
    -------
        :obj:`etree._Element`
            Instance of `etree._Element` corresponding to the newly created element.
    """
    # Split the xpath to get the intermediate nodes as a list
    xpath_list = xpath.split('/')
    n = len(xpath_list)

    # Loop over the elements in the XPath from tip to root until the XPath is found to already exist
    elem = None
    i = 0
    for i in range(0, n - 1):
        xpath = '/'.join(xpath_list[0:(n - i)])
        try:
            elems = tree.xpath(xpath)
            if len(elems):
                elem = elems[0]
                break
        except etree.XPathError:
            raise ValueError('Specified XPath {} is invalid.'.format(xpath))

    # If no existing element was found the root elements of the tree and XPath don't match
    if elem is None:
        raise ValueError("Specified XPath is incompatible with the given XML tree: root tags don't match")

    # Loop over the part of the XPath beyond this point and create all intermediate elements including attributes
    for j in range(n - i, n):
        tag = xpath_list[j]

        # See if this node has an integer index specified
        match_ind = re_ind.search(tag)
        if match_ind:
            tag = tag[:match_ind.start()] + tag[match_ind.end():]
            index = int(match_ind.group(1)) - 1
        else:
            index = 0

        # Find any attributes at this node
        attrib = {}
        match_attr = list(re_atr.finditer(tag))
        if match_attr:
            # Loop over all attributes on this node
            for match in match_attr:
                if match.start() < len(tag):
                    tag = tag[:match.start()]
                attrib.update({match.group(1): match.group(2)})

        # Check if there are siblings with the same name
        siblings = elem.findall(tag)
        n_siblings = len(siblings)

        # Check if there's a sibling with the same name at this index without conflicting attributes
        if index < n_siblings and not any(
                [siblings[index].attrib[key] != attrib[key] for key in attrib.keys() if key in siblings[index].attrib]):
            # If so, use it instead of adding a new one
            siblings[index].attrib.update(attrib)
            elem = siblings[index]
        elif index <= n_siblings:
            # In this case just append a new element
            _elem = etree.Element(tag, attrib)
            elem.append(_elem)
            elem = _elem
        else:
            # In the last case, insert as many empty siblings until this node's index
            sibling = None
            for i in range(index - n_siblings):
                _sibling = etree.Element(tag)
                if i == 0:
                    if not n_siblings:
                        elem.append(_sibling)
                    else:
                        siblings[-1].addnext(_sibling)
                else:
                    sibling.addnext(_sibling)
                sibling = _sibling

            # Finally at a new element at the right index with all attributes
            elem = etree.Element(tag, attrib)
            sibling.addnext(elem)

        # Finally we can update the current XPath, since it has been assured to exist at this point
        xpath = '/'.join([xpath, xpath_list[j]])

    # If a value was supplied assign it to the deepest element in the XPath
    if value is not None:
        value_to_xml(elem, value)

    return elem


def xml_merge(base, merger, out_file=None):
    # type: (Union[str, etree._ElementTree], Union[str, etree._ElementTree], Optional[str]) -> None
    """Merge an XML file into another.

    First two parameters can be either a path to an XML file or an instance of `etree._ElementTree` corresponding to an
    XML tree. All content from the merger will be merger into the base. The third parameter is optional. If set, the
    result of the merger will be written to the file at this path.

    This function does not return anything. If  base is an instance of `etree._ElementTree` this object will be changed,
    if it is a `str` the file at that location will be changed. However, if ``out_file`` is set, the file at that
    location will be changed instead, and not the one at base.

    Parameters
    ----------
        base : str or :obj:`etree._ElementTree`
            Path to or `etree._ElementTree` of an XML file into which the merger should be merged.

        merger : str or :obj:`etree._ElementTree`
            Path to or 'etree._ElementTree` of an XML file which should be merged into the base.

        out_file : str, optional
            Path to a file into which the result of the merger should be written. If not given, the result will
            overwrite the base.

    Notes
    -----
        If conflicting elements exist the value of the merger will overwrite the one in the base.
    """
    if isinstance(base, str):
        try:
            doc = etree.parse(base, parser)
        except IOError:
            if out_file is None:
                out_file = base

            if isinstance(merger, str):
                copyfile(merger, out_file)
            else:
                merger.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)

            return
    else:
        doc = base

    merger_dict = xml_to_dict(merger)
    for xpath, value in merger_dict.items():
        xml_safe_create_element(doc, xpath, value)

    if out_file is not None:
        doc.write(out_file, encoding='utf-8', pretty_print=True, xml_declaration=True)
    elif isinstance(base, str):
        doc.write(base, encoding='utf-8', pretty_print=True, xml_declaration=True)
