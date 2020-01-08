#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2018 D. de Vries and I. van Gent

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains the definition of the `CMDOWSObject` class.
"""
from __future__ import absolute_import, division, print_function

import os

from lxml import etree
from lxml.etree import _Element
from typing import Dict, List, Union

from openlego.utils.cmdows_utils import get_loop_nesting_obj, get_element_by_uid, \
    get_doe_setting_safe
from typing import Any, Optional
from cached_property import cached_property

from openlego.utils.xml_utils import xpath_to_param
from .discipline_resolver import DisciplineResolver


class InvalidCMDOWSFileError(ValueError):

    def __init__(self, reason=None):
        msg = 'Invalid CMDOWS file'
        if reason is not None:
            msg += ': {}'.format(reason)
        super(InvalidCMDOWSFileError, self).__init__(msg)


class CMDOWSObject(object):
    """A class that depends on a CMDOWS file and a Knowledge Base.

    This class is used as a mixin for the `LEGOModel` and `LEGOProblem` classes to avoid code
    dependency.

    Attributes
    ----------
        cmdows_path
        kb_path
        driver_uid

        data_folder : str, optional
            Path to the folder in which to store all data generated during the `Problem` execution.

        base_xml_file : str, optional
            Path to an XML file which should be kept up-to-date with the latest data describing the
            problem.
    """

    SUPERDRIVER_PREFIX = '__SuperDriverComponent__'
    SUBDRIVER_PREFIX = '__SubDriverComponent__'
    SUPERCOMP_PREFIX = '__SuperComponent__'

    def __init__(self, cmdows_path=None, kb_path='', driver_uid=None, data_folder=None,
                 base_xml_file=None, discipline_resolvers=None, keep_files=False, **kwargs):
        # type: (Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], Optional[list], bool, dict) -> None
        """Initialize CMDOWS dependent class from given CMDOWS file, knowledge base and driver UID.

        It is also possible to specify where (temporary) data should be stored, and if a base XML
        file should be kept up-to-data.

        Parameters
        ----------
            cmdows_path : str, optional
                Path to the CMDOWS file.

            kb_path : str, optional
                Path to the knowledge base.

            driver_uid : str, optional
                UID of the main driver under consideration.

            data_folder : str, optional
                Path to the data folder in which to store all files and output from the problem.

            base_xml_file : str, optional
                Path to a base XML file to update with the problem data.
        """
        self._cmdows_path = cmdows_path
        self._kb_path = kb_path
        self._driver_uid = driver_uid
        self.data_folder = data_folder
        self.keep_files = keep_files
        self.base_xml_file = base_xml_file
        self._super_driver_components = []

        self._discipline_resolvers = discipline_resolvers or []  # type: List[DisciplineResolver]

        super(CMDOWSObject, self).__init__()

        if self._driver_uid is None:
            super_drivers = self.__check_for_super_driver()
            if len(super_drivers) == 1:
                self._driver_uid = super_drivers[0]
            elif len(super_drivers) > 1:
                self._driver_uid = None
                self._super_driver_components = super_drivers

    def __getattribute__(self, name):
        # type: (str) -> Any
        """Check the integrity before returning any of the cached variables.

        Parameters
        ----------
        name : str
            Name of the attribute to read.

        Returns
        -------
            any
                The value of the requested attribute.
        """
        if name != '__class__' and name != '__dict__':
            if name in [_name for _name, val in self.__class__.__dict__.items() if
                        isinstance(val, cached_property)]:
                self.__integrity_check()
        return super(CMDOWSObject, self).__getattribute__(name)

    def __setattr__(self, name, value):
        # type: (str, Any) -> None
        """Add check on data_folder when setting it.

        Parameters
        ----------
            name : str
                Name of the attribute.

            value : any
                Value to set the attribute to.
        """
        if name == 'data_folder':
            self.__data_folder_check(value)

        super(CMDOWSObject, self).__setattr__(name, value)

    def __integrity_check(self):
        # type: () -> None
        """Ensure a CMDOWS file has been supplied.

        Raises
        ------
            ValueError
                If no CMDOWS file has been supplied
        """
        if self._cmdows_path is None:
            raise ValueError('No CMDOWS file specified!')

    @staticmethod
    def __data_folder_check(folder_string):
        # type: () -> None
        """Ensure that the data folder for output files exists or else create it."""
        if folder_string:
            if not os.path.exists(folder_string):
                os.makedirs(folder_string)

    def __check_for_super_driver(self):
        # type: () -> List[str]
        """:obj:`List[str]`: List with the super drivers in the CMDOWS file."""
        return self._get_super_drivers(self.full_loop_nesting)

    def _get_super_drivers(self, loop_nesting_obj):
        # type: (List[str, dict]) -> List[str]
        """:obj:`List[str]`: List with the super drivers in the CMDOWS file."""
        super_drivers = []
        for item in loop_nesting_obj:
            if isinstance(item, dict):
                loop_elem_name = list(item)[0]
                if self.loop_element_types[loop_elem_name] in ['optimizer', 'doe']:
                    super_drivers.append(loop_elem_name)
                else:
                    super_drivers.extend(self._get_super_drivers(item[list(item)[0]]))
        return super_drivers

    @cached_property
    def super_drivers(self):
        # type: () -> List[str]
        """:obj:`List[str]`: List with the superdrivers in the CMDOWS file."""
        return self._get_super_drivers(self.full_loop_nesting)

    @cached_property
    def sub_drivers(self):
        # type: () -> List[str]
        """:obj:`List[str]`: List with the subdrivers in the CMDOWS file."""
        return self._get_sub_drivers(self.full_loop_nesting)

    def _get_sub_drivers(self, loop_nesting_obj, super_driver_encountered=False):
        # type: (List[str, dict], bool) -> List[str]
        """Function to determine the subdrivers in a CMDOWS file.

        Parameters
        ----------
            loop_nesting_obj : List[dict, str]
                Object containing the way different elements are nested.

            super_driver_encountered : bool
                Boolean on whether the superdriver has been encountered (used for iterative use of
                this function).

        Returns
        -------
            sub_drivers : List[str]
                List with the subdrivers that were found."""
        sub_drivers = []
        for item in loop_nesting_obj:
            if isinstance(item, dict):
                loop_elem_name = list(item)[0]
                if self.loop_element_types[loop_elem_name] in ['optimizer', 'doe'] and \
                        not super_driver_encountered:
                    sub_drivers.extend(self._get_sub_drivers(item[list(item)[0]], True))
                elif self.loop_element_types[loop_elem_name] in ['optimizer', 'doe'] and \
                        super_driver_encountered:
                    sub_drivers.append(loop_elem_name)
                    sub_drivers.extend(self._get_sub_drivers(item[list(item)[0]], True))
                else:
                    sub_drivers.extend(self._get_sub_drivers(item[list(item)[0]]))
        return sub_drivers

    @cached_property
    def coordinators(self):
        # type: () -> List[str]
        """:obj:`List[str]`: List with the coordinators in the CMDOWS file."""
        _coordinators = []
        for elem in self.elem_arch_elems.iterfind('executableBlocks/coordinators/coordinator'):
            _coordinators.append(elem.attrib['uID'])
        return _coordinators

    def invalidate(self):
        # type: () -> None
        """Invalidate the instance.

        All computed (cached) properties will be recomputed upon being read once the instance has
        been invalidated."""
        __dict__ = self.__class__.__dict__.copy()
        __dict__.update(CMDOWSObject.__dict__)
        for name, value in __dict__.items():
            if isinstance(value, cached_property):
                if name in self.__dict__:
                    del self.__dict__[name]

    @property
    def cmdows_path(self):
        # type: () -> str
        """:obj:`str`: Path to the CMDOWS file this class corresponds to.

        When this property is set the instance is automatically invalidated.
        """
        return self._cmdows_path

    @cmdows_path.setter
    def cmdows_path(self, cmdows_path):
        # type: (str) -> None
        self._cmdows_path = cmdows_path
        self.invalidate()

    @property
    def kb_path(self):
        # type: () -> str
        """:obj:`str`: Path to the knowledge base.

        When this property is set the instance is automatically invalidated.
        """
        return self._kb_path

    @kb_path.setter
    def kb_path(self, kb_path):
        # type: (str) -> None
        self._kb_path = kb_path
        self.invalidate()

    @property
    def discipline_resolvers(self):
        # type: () -> List[DisciplineResolver]
        return self._discipline_resolvers

    def register_discipline_resolver(self, discipline_resolver, last=False):
        # type: (DisciplineResolver, bool) -> None
        """Register a DisciplineResolver only for this LEGOModel instance."""
        if last:
            self._discipline_resolvers.append(discipline_resolver)
        else:
            self._discipline_resolvers.insert(0, discipline_resolver)

    @property
    def driver_uid(self):
        # type: () -> str
        """:obj:`str`: UID of the main driver (to distinguis between main model and subdrivers).

        When this property is set the instance is automatically invalidated.
        """
        return self._driver_uid

    @driver_uid.setter
    def driver_uid(self, driver_uid):
        # type: (str) -> None
        self._driver_uid = driver_uid
        self.invalidate()

    @cached_property
    def elem_cmdows(self):
        # type: () -> _Element
        """:obj:`etree._Element`: Root element of the CMDOWS XML file."""
        return etree.parse(self.cmdows_path).getroot()

    @cached_property
    def elem_problem_def(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The problemDefition element of the CMDOWS file."""
        return self.elem_cmdows.find('problemDefinition')

    @cached_property
    def elem_workflow(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The workflow element of the CMDOWS file."""
        return self.elem_cmdows.find('workflow')

    @cached_property
    def elem_params(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The problemRoles/parameters element of the CMDOWS file."""
        params = self.elem_cmdows.find('problemDefinition/problemRoles/parameters')
        if params is None:
            raise InvalidCMDOWSFileError('does not contain (valid) parameters in the problemRoles')
        return params

    @cached_property
    def elem_arch_elems(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The architectureElements element of the CMDOWS file."""
        arch_elems = self.elem_cmdows.find('architectureElements')
        if arch_elems is None:
            raise InvalidCMDOWSFileError('does not contain (valid) architecture elements')
        return arch_elems

    @cached_property
    def elem_loop_nesting(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The loopNesting element of the CMDOWS file."""
        loop_nesting = self.elem_workflow.find('processGraph/metadata/loopNesting')
        if loop_nesting is None:
            raise InvalidCMDOWSFileError('does not contain loopNesting element')
        return loop_nesting

    @cached_property
    def elem_model_driver(self):
        # type: () -> Union[_Element, None]
        """:obj:`etree._Element`: The XML element of the super driver."""
        uid = self.driver_uid
        if uid:
            elem = get_element_by_uid(self.elem_cmdows, uid)
            if isinstance(elem, _Element):
                return elem
            else:
                raise InvalidCMDOWSFileError('does not contain element with UID {}'.format(uid))
        else:
            return None

    @cached_property
    def full_loop_nesting(self):
        # type: () -> List[str, dict]
        """:obj:`dict`: Dictionary of the loopNesting XML element."""
        return get_loop_nesting_obj(self.elem_loop_nesting)

    @cached_property
    def filtered_loop_nesting(self):
        # type: () -> List[str, dict]
        """:obj:`List[str, dict]`: Filtered loop nesting object.
        """
        return self.filter_loop_nesting(self.full_loop_nesting)

    def filter_loop_nesting(self, loop_nesting_list, sub_driver_found=False):
        # type: (List[str, dict]) -> List[str, dict]
        """This method filters the loop nesting to only return a loop nesting object with entries
        that are relevant for the driver under consideration.

        Parameters
        ----------
        loop_nesting_list : List[dict, str]
            The object specifying the nesting of functions in the CMDOWS file.
        sub_driver_found : bool
            Boolean on whether the subdriver has been found (used to iteratively run this function)

        Returns
        -------
        _filt_loop_nesting_lst : List[str, dict]
            Filtered version of the loop nesting based on the current driver under consideration.
        """
        _filt_loop_nesting_lst = []
        if self._driver_uid in self.sub_drivers and not sub_driver_found:
            add_all_blocks = False
        else:
            add_all_blocks = True
        driver_uid_is_sub_driver = True if self._driver_uid in self.sub_drivers else False
        for item in loop_nesting_list:
            if isinstance(item, dict):
                loop_el = list(item)[0]
                if self.loop_element_types[loop_el] == 'converger' and add_all_blocks:
                    _filt_loop_nesting_lst.append(item)
                elif self.loop_element_types[loop_el] == 'coordinator':
                    _filt_loop_nesting_lst\
                        .append({loop_el: self.filter_loop_nesting(item[loop_el])})
                elif self.loop_element_types[loop_el] in ['optimizer', 'doe']:
                    if loop_el == self._driver_uid:
                        if driver_uid_is_sub_driver:
                            _filt_loop_nesting_lst\
                                .append({loop_el: self.filter_loop_nesting(item[loop_el],
                                                                           sub_driver_found=True)})
                        else:
                            _filt_loop_nesting_lst\
                                .append({loop_el: self.filter_loop_nesting(item[loop_el],
                                                                           sub_driver_found=False)})
                    else:
                        if driver_uid_is_sub_driver:
                            if loop_el in self.sub_drivers:
                                pass
                            else:
                                _filt_loop_nesting_lst.append(
                                    {self.SUPERDRIVER_PREFIX + loop_el:
                                         self.filter_loop_nesting(item[loop_el],
                                                                  sub_driver_found=False)})
                        else:
                            _filt_loop_nesting_lst.append(self.SUBDRIVER_PREFIX + loop_el)
                elif self.loop_element_types[loop_el] == 'distributed_system_converger':
                    # Find the right sublist of the multiple subworkflows handled by the converger
                    wf_to_append = []
                    driver_key = None
                    for idx, subworkflow in enumerate(item[loop_el]):
                        if not driver_uid_is_sub_driver:
                            if list(subworkflow)[0] == self.driver_uid:
                                wf_to_append = subworkflow[self.driver_uid]
                                driver_key = self.driver_uid
                                break
                            elif list(subworkflow)[0] in self._super_driver_components:
                                driver_key = loop_el
                                wf_to_append.append(list(subworkflow)[0])
                        else:
                            entry_to_check = subworkflow[list(subworkflow)[0]][0]
                            if isinstance(entry_to_check, dict) and \
                                    list(entry_to_check)[0] == self.driver_uid:
                                wf_to_append = subworkflow[list(subworkflow)[0]]
                                driver_key = self.SUPERDRIVER_PREFIX + list(subworkflow)[0]
                                break
                    if wf_to_append is None or driver_key is None:
                        raise AssertionError('Could not match the right subworkflow for '
                                             'driver_uid: {}'.format(self.driver_uid))
                    if all(isinstance(elem, str) for elem in wf_to_append):
                        _filt_loop_nesting_lst.append({driver_key: wf_to_append})
                    else:
                        _filt_loop_nesting_lst.\
                            append({driver_key: self.filter_loop_nesting(wf_to_append)})
                else:
                    raise AssertionError('Could not find element details for loop element {}.'
                                         .format(loop_el))
            elif isinstance(item, str):
                if add_all_blocks:
                    _filt_loop_nesting_lst.append(item)
                else:
                    pass
            else:
                raise AssertionError('Invalid type {} found in loop nesting object.'
                                     .format(type(item)))
        return _filt_loop_nesting_lst

    @cached_property
    def model_exec_blocks(self):
        # type: () -> List[str]
        """:obj:`List[str]`: List of all executable blocks in the model w.r.t. the driver UID."""
        return self.collect_all_exec_blocks(self.filtered_loop_nesting)

    @cached_property
    def model_nested_exec_blocks(self):
        """:obj:`List[str]`: List of all the executable blocks that are nested inside  w.r.t. the
        superdriver UID."""
        if self.driver_uid in self.super_drivers or self._super_driver_components:
            return self.collect_all_exec_blocks(self.full_loop_nesting,
                                                nested_in=self.driver_uid)
        else:
            return []

    def collect_all_exec_blocks(self, loop_nesting_list, nested_in=None):
        # type: (List[str, dict]) -> List[str]
        """Method collects the executable blocks in the model w.r.t. the driver UID under
        consideration.

        Parameters
        ----------
        loop_nesting_list : List[dict, str]
            The loop nesting object
        nested_in : Union[None, str]
            UID of executable block in which the blocks should be nested

        Returns
        -------
        all_executable_blocks : List[str]
            List with all relevant executable blocks
        """
        all_executable_blocks = []
        break_loop = False
        for item in loop_nesting_list:
            if isinstance(item, dict):
                loop_elem_name = list(item)[0]
                if nested_in is not None and loop_elem_name == nested_in:
                    nested_in = None
                    break_loop = True
                all_executable_blocks.extend(self.collect_all_exec_blocks(item[loop_elem_name],
                                                                          nested_in=nested_in))
                if break_loop:
                    break
            elif isinstance(item, str):
                if not nested_in:
                    all_executable_blocks.append(item)
        return all_executable_blocks

    @cached_property
    def all_loop_elements(self):
        # type: () -> List[str]
        """:obj:`List[str]`: Lists all loop elements w.r.t. the driver UID under consideration."""
        return self.collect_all_loop_elements(self.filtered_loop_nesting)

    def collect_all_loop_elements(self, loop_nesting_list):
        # type: (List[str, dict]) -> List[str]
        """Method collects the loop elements from a loop nesting object.

        Parameters
        ----------
        loop_nesting_list : List[dict, str]
            The loop nesting object

        Returns
        -------
        all_loop_elements : List[str]
            The loop elements in the loop nesting object
        """

        all_loop_elements = []
        for item in loop_nesting_list:
            if isinstance(item, dict):
                loop_elem_name = list(item)[0]
                all_loop_elements.append(loop_elem_name)
                all_loop_elements.extend(self.collect_all_loop_elements(item[loop_elem_name]))
        return all_loop_elements

    @cached_property
    def has_optimizer(self):
        # type: () -> bool
        """:obj:`bool`: True if there is an optimizer, False if not."""
        if self.driver_uid:
            if self.loop_element_types[self.driver_uid] == 'optimizer':
                return True
        return False

    @cached_property
    def has_doe(self):
        # type: () -> bool
        """:obj:`bool`: True if there is a DOE component, False if not."""
        if self.driver_uid:
            if self.loop_element_types[self.driver_uid] == 'doe':
                return True
        return False

    @cached_property
    def has_driver(self):
        # type: () -> bool
        """:obj:`bool`: True if there is a driver component (DOE or optimizer), False if not."""
        if self.has_doe or self.has_optimizer:
            return True
        return False

    @cached_property
    def doe_uids(self):
        # type: () -> List[str]
        """List of UIDs of the DOE elements."""
        return [x for x in self.loop_element_types if self.loop_element_types[x] == 'doe']

    @cached_property
    def distributed_system_converger_uids(self):
        # type: () -> List[str]
        """List of UIDs of the distributed system converger elements."""
        return [x for x in self.loop_element_types if
                self.loop_element_types[x] == 'distributed_system_converger']

    @cached_property
    def has_distributed_system_converger(self):
        # type: () -> bool
        """:obj:`bool`: True if there is a distributed system converger element in the CMDOWS file,
        False if not."""
        return bool(self.distributed_system_converger_uids)

    @cached_property
    def doe_runs(self):
        # type: () -> Dict[str,float]
        """Dictionary containing mapping between DOE UID and number of runs."""
        _doe_runs_dict = {}
        for doe_uid in self.doe_uids:
            doe_elem = get_element_by_uid(self.elem_cmdows, doe_uid)
            # Load settings from CMDOWS file
            doe_method = get_doe_setting_safe(doe_elem, 'method', 'Uniform design')  # type: str
            doe_runs = get_doe_setting_safe(doe_elem, 'runs', None, expected_type='int',
                                            doe_method=doe_method,
                                            required_for_doe_methods=['Latin hypercube design',
                                                                      'Uniform design',
                                                                      'Monte Carlo design'])
            _doe_runs_dict.update({doe_uid: doe_runs})
        return _doe_runs_dict

    @cached_property
    def doe_parameters(self):
        # type: () -> Dict[str]
        """Mapping between all input and output parameters of all DOE element and their size."""
        _doe_parameters = {}
        for doe_uid in self.doe_uids:
            for inp in self.doe_samples[doe_uid]['inputs']:
                _doe_parameters[inp] = {'size': self.doe_runs[doe_uid]}
            for out in self.doe_samples[doe_uid]['outputs']:
                _doe_parameters[out] = {'size': self.doe_runs[doe_uid]}
        return _doe_parameters

    @cached_property
    def doe_samples(self):
        # type: () -> Dict[str]
        """DOE element UIDs and their input and output parameters containing samples."""
        _doe_samples = {}
        for doe_uid in self.doe_uids:
            _doe_samples[doe_uid] = entry = {}
            entry.update({'inputs': self.get_source_parameters(doe_uid,
                                                               self.doe_sample_lists['inputs'])})
            entry.update({'outputs': self.get_target_parameters(doe_uid,
                                                                self.doe_sample_lists['outputs'])})
        return _doe_samples

    @cached_property
    def doe_sample_lists(self):
        # type: () -> Dict[str]
        """Input and output sample lists stored in the CMDOWS file under architecture elements."""
        _doe_sample_lists = {'inputs': [], 'outputs': []}
        for elem in self.elem_arch_elems.iter('doeInputSampleList'):
            _doe_sample_lists['inputs'].append(elem.attrib['uID'])
        for elem in self.elem_arch_elems.iter('doeOutputSampleList'):
            _doe_sample_lists['outputs'].append(elem.attrib['uID'])
        return _doe_sample_lists

    @cached_property
    def sm_uids(self):
        # type: () -> List[str]
        return [sm.attrib['uID'] for sm in self.elem_cmdows.iter('surrogateModel')]

    @cached_property
    def sm_prediction_inputs(self):
        # type: () -> Dict[str]
        return self._get_sm_parameters('prediction/inputs/input/parameterUID')

    @cached_property
    def sm_prediction_outputs(self):
        # type: () -> Dict[str]
        return self._get_sm_parameters('prediction/outputs/output/parameterUID')

    @cached_property
    def sm_training_inputs(self):
        # type: () -> Dict[str]
        return self._get_sm_parameters('training/inputs/input/parameterUID')

    @cached_property
    def sm_training_outputs(self):
        # type: () -> Dict[str]
        return self._get_sm_parameters('training/outputs/output/parameterUID')

    @cached_property
    def sm_of_training_params(self):
        # type: () -> Dict[str]
        """Mapping between training parameters and surrogate model UIDs."""
        _sm_mapping = {}
        for sm_uid, xpaths in self.sm_training_inputs.items():
            for xpath in xpaths:
                _sm_mapping[xpath_to_param(xpath)] = sm_uid
        for sm_uid, xpaths in self.sm_training_outputs.items():
            for xpath in xpaths:
                _sm_mapping[xpath_to_param(xpath)] = sm_uid
        return _sm_mapping

    def _get_sm_parameters(self, xpath):
        # type: (str) -> Dict[str]
        """Retrieve surrogate model parameters meeting a certain XPath.

        Parameters
        ----------
            xpath : str
                XPath to be searched for.
        """
        _sm_parameters = {}
        for elem_sm in self.elem_cmdows.iter('surrogateModel'):
            sm_uid = elem_sm.attrib['uID']
            if sm_uid not in list(_sm_parameters):
                _sm_parameters[sm_uid] = set()
            for el_parameter in elem_sm.iterfind(xpath):
                _sm_parameters[sm_uid].update([el_parameter.text])
        return _sm_parameters

    @cached_property
    def block_order(self):
        # type: () -> List[str]
        """:obj:`List[str]`: Order of all blocks based on the step_numbers provided in CMDOWS file.
        processGraph."""
        return [x for _, x in sorted(zip(self.process_info['step_numbers'],
                                         self.process_info['uids']))]

    @cached_property
    def partition_sets(self):
        # type: () -> Dict[int, set]
        """:obj:`dict` of :obj:`set`: Dictionary of executable block ``uIDs`` per partitionID."""
        partitions = dict()
        for idx, block in enumerate(self.process_info['uids']):
            if self.process_info['partition_ids'][idx] is not None:
                partition_id = self.process_info['partition_ids'][idx]
                if partition_id not in partitions:
                    partitions[partition_id] = {block}
                else:
                    partitions[partition_id].add(block)
        return partitions

    @cached_property
    def coupled_blocks(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: Lists ``uIDs`` of the coupled executable blocks specified in
        CMDOWS file."""
        _coupled_blocks = []
        for block in self.elem_arch_elems.iterfind(
                'executableBlocks/coupledAnalyses/coupledAnalysis/relatedExecutableBlockUID'):
            if block.text in self.model_exec_blocks:
                _coupled_blocks.append(block.text)
        return _coupled_blocks

    @cached_property
    def loop_element_types(self):
        # type: () -> Dict[str]
        """:obj:`dict` of :obj:`str`: Mapping of loop elements specified in CMDOWS file."""
        _loopelement_details = {}
        for elem in self.elem_arch_elems.iterfind('executableBlocks/coordinators/coordinator'):
            _loopelement_details[elem.attrib['uID']] = 'coordinator'
        for elem in self.elem_arch_elems.iterfind('executableBlocks/convergers/converger'):
            _loopelement_details[elem.attrib['uID']] = 'converger'
        for elem in self.elem_arch_elems.iterfind('executableBlocks/optimizers/optimizer'):
            _loopelement_details[elem.attrib['uID']] = 'optimizer'
        for elem in self.elem_arch_elems.iterfind('executableBlocks/does/doe'):
            _loopelement_details[elem.attrib['uID']] = 'doe'
        for elem in self.elem_arch_elems.iterfind('executableBlocks/distributedSystemConvergers/'
                                                  'distributedSystemConverger'):
            _loopelement_details[elem.attrib['uID']] = 'distributed_system_converger'
        return _loopelement_details

    @cached_property
    def process_info(self):
        # type: () -> dict
        """:obj:`dict`: Dictionary with information from the processGraph element."""
        _uids = []
        _process_step_numbers = []
        _partition_ids = []
        for elem in self.elem_workflow.iterfind('processGraph/nodes/node'):
            _uids.append(elem.find('referenceUID').text)
            _process_step_numbers.append(int(elem.find('processStepNumber').text))
            if elem.find('partitionID') is not None:
                _partition_ids.append(elem.find('partitionID').text)
            else:
                _partition_ids.append(None)
        return {'uids': _uids, 'step_numbers': _process_step_numbers,
                'partition_ids': _partition_ids}

    @cached_property
    def block_step_numbers(self):
        # type: () -> Dict[int]
        """:obj:`Dict[int]`: Mapping between block UID (key) and process step number (value)."""
        _process_step_numbers = {}
        for i, uid in enumerate(self.process_info['uids']):
            _process_step_numbers[uid] = self.process_info['step_numbers'][i]
        return _process_step_numbers

    def get_source_functions(self, parameter_uid, filter_list=None):
        # type: (str, List[str]) -> List[str]
        """
        Get the source functions of a certain parameter UID in the CMDOWS file based on the
        workflow/dataGraph element.

        Parameters
        ----------
            parameter_uid : str
                UID of the parameter to analyze
            filter_list : List[str]
                list with target functions to include

        Returns
        -------
            list with source functions
        """
        _srce_funcs = [x for x in self.elem_cmdows.xpath(r'workflow/dataGraph/edges/edge'
                                                         r'[toParameterUID="{}"]/fromExecutable'
                                                         r'BlockUID/text()'.format(parameter_uid))]
        if filter_list is not None:
            return list(filter(lambda y: y in filter_list, _srce_funcs))
        else:
            return _srce_funcs

    def get_target_functions(self, parameter_uid, filter_list=None):
        # type: (str, List[str]) -> List[str]
        """
        Get the target functions of a certain parameter UID in the CMDOWS file based on the
        workflow/dataGraph element.

        Parameters
        ----------
            parameter_uid : str
                UID of the parameter to analyze
            filter_list : List[str]
                list with target functions to include

        Returns
        -------
            list with target functions
        """
        _trgt_funcs = [x for x in self.elem_cmdows.xpath(r'workflow/dataGraph/edges/edge'
                                                         r'[fromParameterUID="{}"]/toExecutable'
                                                         r'BlockUID/text()'.format(parameter_uid))]
        if filter_list is not None:
            return list(filter(lambda y: y in filter_list, _trgt_funcs))
        else:
            return _trgt_funcs

    def get_source_parameters(self, function_uid, filter_list=None):
        # type: (str) -> List[str]
        """
        Get the source parameters of a certain function UID in the CMDOWS file based on the
        workflow/dataGraph element.

        Parameters
        ----------
            function_uid : str
                UID of the function to analyze
            filter_list : List[str]
                list with source parameters to include

        Returns
        -------
            list with target parameters
        """
        _srce_parms = [x for x in self.elem_cmdows.xpath(r'workflow/dataGraph/edges/edge[to'
                                                         r'ExecutableBlockUID="{}"]/fromParameter'
                                                         r'UID/text()'.format(function_uid))]
        if filter_list is not None:
            return list(filter(lambda y: y in filter_list, _srce_parms))
        else:
            return _srce_parms

    def get_target_parameters(self, function_uid, filter_list=None):
        # type: (str) -> List[str]
        """
        Get the target parameters of a certain function UID in the CMDOWS file based on the
        workflow/dataGraph element.

        Parameters
        ----------
            function_uid : str
                UID of the function to analyze
            filter_list : List[str]
                list with target parameters to include

        Returns
        -------
            list with target parameters
        """
        _trgt_parms = [x for x in self.elem_cmdows.xpath(r'workflow/dataGraph/edges/edge[from'
                                                         r'ExecutableBlockUID="{}"]/toParameterUID'
                                                         r'/text()'.format(function_uid))]
        if filter_list is not None:
            return list(filter(lambda y: y in filter_list, _trgt_parms))
        else:
            return _trgt_parms
