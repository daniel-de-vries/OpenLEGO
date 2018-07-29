from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import copy
from lxml.etree import _Element


# TODO: add function signature description
# TODO: add function docstring
def get_related_parameter_uid(elem_param, full_xml):

    param = elem_param.attrib['uID']
    if isinstance(elem_param.find('relatedParameterUID'), _Element):
        mapped = elem_param.find('relatedParameterUID').text
    elif isinstance(elem_param.find('relatedInstanceUID'), _Element):
        related_instance = get_element_by_uid(full_xml, elem_param.find('relatedInstanceUID').text)
        mapped = related_instance.find('relatedParameterUID').text
    else:
        raise AssertionError('Could not map element {}.'.format(param))
    return param, mapped


def get_element_by_uid(xml, uid):
    # type: (etree._ElementTree, str) -> etree._ElementTree
    """Method to get the element based on a UID value.

    Parameters
    ----------
        xml : :obj:`etree._ElementTree`
            `etree._ElementTree` of an XML file where the uID should be found.

        uid : str
            uID to be found.

    """
    xpath_expression = get_uid_search_xpath(uid)
    els = xml.xpath(xpath_expression)
    if len(els) > 1:
        raise AssertionError('Multiple elements with UID ' + uid + ' found. Use "check_uids()" to check if all UIDs'
                                                                   ' are unique.')
    elif len(els) == 0:
        raise AssertionError('Could not find element with UID ' + uid + '.')
    return els[0]


# TODO: add function signature description
# TODO: use correct docstring format
def get_uid_search_xpath(uid):
    """ Method to get the XPath expression for a uID which might contain quote characters.

    # TODO: update docstring

    :param uid: uID
    :type uid: str
    :return: XPath expression
    :rtype: str
    """
    if '"' in uid or '&quot;' in uid:
        uid_concat = "concat('%s')" % uid.replace('&quot;', "\',\'\"\',\'").replace('"', "\',\'\"\',\'")
        return './/*[@uID=' + uid_concat + ']'
    else:
        return './/*[@uID="' + uid + '"]'


# TODO: add function signature description
# TODO: use correct docstring format
def get_loop_nesting_dict(r):
    """Function to make a dictionary of the loop hierarchy based on the loopNesting element in a CMDOWS file.

    :param r:
    :type r:
    :return:
    :rtype:
    """
    basic_list = []
    basic_dict = {}
    d = copy(r.attrib)
    if r.tag == 'loopNesting' or r.tag == 'loopElements' or r.tag == 'functionElements':
        for x in r.iterchildren():
            basic_list.append(get_loop_nesting_dict(x))
        if len(basic_list) == 1 and isinstance(basic_list[0], list):
            return basic_list[0]
        else:
            return basic_list
    elif r.tag == 'loopElement':
        basic_dict[d['relatedUID']] = []
        for x in r.iterchildren():
            basic_dict[d['relatedUID']].extend(get_loop_nesting_dict(x))
        return basic_dict
    elif r.tag == 'functionElement':
        return r.text
    else:
        raise AssertionError('Something went wrong in this function...')