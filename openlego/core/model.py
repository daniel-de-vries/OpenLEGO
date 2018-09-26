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

This file contains the definition of the `LEGOModel` class.
"""
from __future__ import absolute_import, division, print_function

import copy
import imp
import os
import warnings

import numpy as np

from cached_property import cached_property
from lxml import etree
from lxml.etree import _Element, _ElementTree
from openmdao.api import Group, IndepVarComp, LinearBlockGS, NonlinearBlockGS, LinearBlockJac, NonlinearBlockJac, \
    LinearRunOnce, ExecComp, NonlinearRunOnce, DirectSolver
from typing import Union, Optional, List, Any, Dict, Tuple

from openlego.utils.general_utils import parse_cmdows_value, str_to_valid_sys_name, parse_string
from openlego.utils.xml_utils import xpath_to_param, xml_to_dict
from openlego.utils.cmdows_utils import get_element_by_uid, get_related_parameter_uid, get_loop_nesting_obj
from .abstract_discipline import AbstractDiscipline
from .cmdows import CMDOWSObject, InvalidCMDOWSFileError
from .discipline_component import DisciplineComponent


class LEGOModel(CMDOWSObject, Group):
    """Specialized OpenMDAO Group class representing the model specified by a CMDOWS file.

    An important note about this class in the context of OpenMDAO is that the aggregation pattern of the root Group
    class the base Problem class has is changed into a stronger composition pattern. This is because this class directly
    controls the creation and assembly of this class by making use of Python's @property decorator. It is not possible,
    nor should it be attempted, to manually inject a different instance of Group  in place of these, because the
    correspondence between the CMDOWS file and the Problem can then no longer be guaranteed.

    Attributes
    ----------
        objective_required
        discipline_components
        mapped_parameters
        mapped_parameters_inv
        mathematical_functions_inputs
        mathematical_functions_outputs
        mathematical_functions_groups
        variables_sizes
        coupling_vars
        coupling_var_copies
        coupling_var_cons
        des_var_copies
        loop_nesting_obj
        loop_nesting_details
        coupled_hierarchy
        model_sub_drivers
        model_super_drivers
        model_super_components
        system_inputs
        model_required_inputs
        model_all_outputs
        model_super_inputs
        model_super_inputs_inv
        model_constants
        model_super_outputs
        design_vars
        constraints
        objective
        system_order
        coordinator
    """

    def __init__(self, cmdows_path=None, kb_path='', driver_uid=None, data_folder=None, base_xml_file=None, **kwargs):
        # type: (Optional[str], Optional[str], Optional[str], Optional[str]) -> None
        """Initialize a CMDOWS Problem from a given CMDOWS file and knowledge base.

        It is also possible to specify where (temporary) data should be stored, and if a base XML
        file should be kept up-to-data.

        Parameters
        ----------
        cmdows_path : str, optional
            Path to the CMDOWS file.

        kb_path : str, optional
            Path to the knowledge base.

        data_folder : str, optional
            Path to the data folder in which to store all files and output from the problem.

        base_xml_file : str, optional
            Path to a base XML file to update with the problem data.
        """
        self.linear_solver = LinearRunOnce()
        self.nonlinear_solver = NonlinearRunOnce()
        super(LEGOModel, self).__init__(cmdows_path, kb_path, driver_uid, data_folder, base_xml_file, **kwargs)

    def __setattr__(self, name, value):
        # type: (str, Any) -> None
        """Bypass setting coordinator and coupled_group attributes.

        Parameters
        ----------
            name : str
                Name of the attribute.

            value : any
                Value to set the attribute to.
        """
        if name not in ['coordinator', 'coupled_group']:
            super(LEGOModel, self).__setattr__(name, value)

    def does_value_fit(self, name, val):
        # type: (str, Union[str, float, np.ndarray]) -> bool
        """Check whether a given value has the correct size to be assigned to a given variable.

        Parameters
        ----------
            name : str
                Name of the variable.

            val : str or float or np.ndarray
                Value to check.

        Returns
        -------
            bool
                `True` if the value fits, `False` if not.
        """
        return (isinstance(val, np.ndarray) and val.size == self.variable_sizes[name]) \
            or (not isinstance(val, np.ndarray) and self.variable_sizes[name] == 1)

    @cached_property
    def objective_required(self):
        # type: () -> bool
        """:obj:`bool`: True if an objective value is required, False if not."""
        if self.has_optimizer:
            return True
        return False

    @cached_property
    def discipline_components(self):
        # type: () -> Dict[str, DisciplineComponent]
        """:obj:`dict`: Dictionary of discipline components by their design competence ``uID`` from CMDOWS.

        Raises
        ------
            RuntimeError
                If a ``designCompetence`` specified in the CMDOWS file does not correspond to an `AbstractDiscipline`.
        """
        # Ensure that the knowledge base path is specified.
        _discipline_components = dict()
        for design_competence in self.elem_cmdows.iter('designCompetence'):
            if design_competence.attrib['uID'] in self.model_exec_blocks or self.driver_uid in self.super_drivers:
                if not self._kb_path or not os.path.isdir(self._kb_path):
                    raise ValueError('No valid knowledge base path ({}) specified while the CMDOWS file contains design'
                                     ' competences.'.format(self._kb_path))
                uid = design_competence.attrib['uID']
                name = design_competence.find('ID').text
                try:
                    fp, pathname, description = imp.find_module(name, [self.kb_path])
                    mod = imp.load_module(name, fp, pathname, description)
                    cls = getattr(mod, name)  # type: AbstractDiscipline.__class__
                    if not issubclass(cls, AbstractDiscipline):
                        raise RuntimeError
                except Exception:
                    raise ValueError(
                        'Unable to process CMDOWS file: no proper discipline found for design competence with name %s'
                        % name)
                finally:
                    if 'fp' in locals():
                        fp.close()

                component = DisciplineComponent(cls(), data_folder=self.data_folder, base_file=self.base_xml_file)
                _discipline_components.update({uid: component})
        return _discipline_components

    @cached_property
    def mapped_parameters(self):
        # type: () -> Dict[str, str]
        """:obj:`dict`: Dictionary of parameters that are mapped in the CMDOWS file, for example as copies."""
        mapped_params = dict()
        for elem_category in self.elem_arch_elems.find('parameters').iterchildren():
            for elem_param in elem_category.iterchildren():
                param, mapped = get_related_parameter_uid(elem_param, self.elem_cmdows)
                mapped_params.update({param: mapped})
        return mapped_params

    @cached_property
    def mapped_parameters_inv(self):
        # type: () -> Dict[str, list]
        """:obj:`dict`: Dictionary with the inverse mapping of the parameters in the CMDOWS file."""
        mapped_params_inv = dict()
        for mapped_param, mapping in self.mapped_parameters.items():
            if mapping not in mapped_params_inv:
                mapped_params_inv[mapping] = [mapped_param]
            else:
                mapped_params_inv[mapping].append(mapped_param)
        return mapped_params_inv

    @cached_property
    def mathematical_functions_inputs(self):
        # type: () -> dict
        """:obj:`dict`: Dictionary of all mathematical function blocks with a list of their input variables."""
        _inputs = dict()
        for mathematical_function in self.elem_cmdows.iter('mathematicalFunction'):
            uid = mathematical_function.attrib['uID']
            local_inputs = list()
            for _input in mathematical_function.iter('input'):
                input_name = xpath_to_param(_input.find('parameterUID').text)
                eq_label = _input.find('equationLabel').text
                local_inputs.append((eq_label, input_name))
            _inputs.update({uid: local_inputs})
        return _inputs

    @cached_property
    def mathematical_functions_outputs(self):
        # type: () -> dict
        """:obj:`dict`: Dictionary of all mathematical function blocks with a list of their output variables."""
        _outputs = dict()
        for mathematical_function in self.elem_cmdows.iter('mathematicalFunction'):
            uid = mathematical_function.attrib['uID']
            local_outputs = list()
            for _output in mathematical_function.iter('output'):
                output_name = xpath_to_param(_output.find('parameterUID').text)
                local_outputs.append(output_name)
            _outputs.update({uid: local_outputs})
        return _outputs

    @cached_property
    def mathematical_functions_groups(self):
        # type: () -> Dict[str, Group]
        """:obj:`dict`: Dictionary of execute components by their mathematical function ``uID`` from CMDOWS.
        """
        _mathematical_functions = dict()
        implemented_eqs = []
        for mathematical_function in self.elem_cmdows.iter('mathematicalFunction'):
            if mathematical_function.attrib['uID'] in self.model_exec_blocks or \
                    self.driver_uid in self.super_drivers:
                uid = mathematical_function.attrib['uID']
                group = Group()

                # First select outputs for which subsystems should actually be added
                required_outputs = []
                for output in mathematical_function.iter('output'):
                    output_name = output.find('parameterUID').text
                    if '/architectureNodes/final' not in output_name:
                        required_outputs.append(output_name)

                # Then create mathematical subsystems for each output
                for output in mathematical_function.iter('output'):
                    if output.find('parameterUID').text in required_outputs:
                        if output.find('equations') is None:
                            eq_uid = output.find('equationsUID')
                            if eq_uid is not None:
                                eqs_elem = get_element_by_uid(self.elem_cmdows, eq_uid.text)
                            else:
                                raise AssertionError('Could not find equation UID.')
                        else:
                            eqs_elem = output.find('equations')
                        for equation in eqs_elem.iter('equation'):
                            if equation.attrib['language'] == 'Python':
                                eq_uid = equation.getparent().attrib['uID']
                                if eq_uid in implemented_eqs:
                                    raise AssertionError('Equation with UID {} is already defined.'.format(eq_uid))
                                implemented_eqs.append(eq_uid)
                                eq_expr = equation.text
                                eq_output = xpath_to_param(output.find('parameterUID').text)
                                eq_output_label = output.find('parameterUID').text.split('/')[-1]
                                if eq_output_label in eq_expr:  # check to avoid use of output name in expression
                                    eq_output_label = eq_output_label + '__output'

                                promotes = list()
                                for eq_label, input_name in self.mathematical_functions_inputs[uid]:
                                    # TODO: This mapping of the input name could get a more sophisticated logic
                                    if input_name in self.mapped_parameters and input_name not in self.design_vars and \
                                            'copyDesignVariable' not in input_name:
                                        input_name = self.mapped_parameters[input_name]

                                    if eq_label in eq_expr:
                                        promotes.append((eq_label, input_name))
                                group.add_subsystem(str_to_valid_sys_name(eq_uid),
                                                    ExecComp(eq_output_label + ' = ' + eq_expr),
                                                    promotes=promotes + [(eq_output_label, eq_output), ])
                _mathematical_functions.update({uid: group})

        return _mathematical_functions

    @cached_property
    def variable_sizes(self):
        # type: () -> Dict[str, int]
        """:obj:`dict`: Dictionary of the sizes of all variables by their names."""
        variable_sizes = {}
        for component in self.discipline_components.values():
            for name, value in component.variables_from_xml.items():
                variable_sizes.update({name: np.atleast_1d(value).size})

        for local_inputs in self.mathematical_functions_inputs.values():
            for _input in local_inputs:
                variable_sizes.update({_input[1]: np.atleast_1d([0]).size})

        for local_outputs in self.mathematical_functions_outputs.values():
            for _output in local_outputs:
                variable_sizes.update({_output: np.atleast_1d([0]).size})

        return variable_sizes

    @cached_property
    def coupling_vars(self):
        # type: () -> Dict[str, Dict[str, str]]
        """:obj:`dict`: Dictionary with coupling variables."""
        coupling_vars = dict()

        # First create a map between related param and coupling copy var
        for var in self.elem_arch_elems.iter('couplingCopyVariable'):
            param, mapped = get_related_parameter_uid(var, self.elem_cmdows)
            coupling_vars.update({xpath_to_param(mapped): xpath_to_param(param)})

        # Then update dict with corresponding consistency constraint var
        for convar in self.elem_arch_elems.iter('consistencyConstraintVariable'):
            param = xpath_to_param(convar.find('relatedParameterUID').text)
            if param not in coupling_vars:
                raise InvalidCMDOWSFileError()
            coupling_vars.update({param: {'copy': coupling_vars[param], 'con': xpath_to_param(convar.attrib['uID'])}})
        return coupling_vars

    @cached_property
    def coupling_var_copies(self):
        # type: () -> Dict[str, str]
        """:obj:`dict`: Dictionary with coupling variable copies."""
        coupling_var_copies = dict()
        for var, value in self.coupling_vars.items():
            coupling_var_copies.update({var: value['copy']})
        return coupling_var_copies

    @cached_property
    def coupling_var_cons(self):
        # type: () -> Optional[Dict[str, str]]
        """:obj:`dict`: Dictionary with coupling variable constraints."""
        coupling_var_cons = dict()
        for var, value in self.coupling_vars.items():
            if isinstance(value, dict):
                if 'con' in value:
                    coupling_var_cons.update({var: value['con']})
        if coupling_var_cons:
            return coupling_var_cons
        else:
            return None

    @cached_property
    def des_var_copies(self):
        # type: () -> Optional[Dict[str, str]]
        """:obj:`dict`: Dictionary with design variable copies."""
        _des_var_copies = dict()
        for var in self.elem_arch_elems.iter('copyDesignVariable'):
            if var.attrib['uID'] in self.design_vars.keys():
                param, mapped = get_related_parameter_uid(var, self.elem_cmdows)
                _des_var_copies.update({xpath_to_param(mapped): xpath_to_param(param)})
        return _des_var_copies

    @cached_property
    def loop_nesting_obj(self):
        # type: () -> Dict[str, dict]
        """:obj:`dict`: Dictionary of the loopNesting XML element."""
        return get_loop_nesting_obj(self.elem_loop_nesting)

    @cached_property
    def loop_element_details(self):
        # type: () -> Dict[str]
        """:obj:`dict` of :obj:`str`: Dictionary with mapping of loop elements specified in the CMDOWS file."""
        _loopelement_details = {}
        for elem in self.elem_arch_elems.iterfind('executableBlocks/coordinators/coordinator'):
            _loopelement_details[elem.attrib['uID']] = 'coordinator'
        for elem in self.elem_arch_elems.iterfind('executableBlocks/convergers/converger'):
            _loopelement_details[elem.attrib['uID']] = 'converger'
        for elem in self.elem_arch_elems.iterfind('executableBlocks/optimizers/optimizer'):
            _loopelement_details[elem.attrib['uID']] = 'optimizer'
        for elem in self.elem_arch_elems.iterfind('executableBlocks/does/doe'):
            _loopelement_details[elem.attrib['uID']] = 'doe'
        return _loopelement_details

    @cached_property
    def coupled_hierarchy(self):
        # type: () -> List[dict]
        """:obj:`list`: List containing the hierarchy of the coupled blocks for grouped convergence."""
        return self._get_coupled_hierarchy(self.full_loop_nesting)

    def _get_coupled_hierarchy(self, hierarchy):
        # type: (List[str, dict]) -> List[dict]
        """:obj:`list`: List containing the partitioned hierarchy of the coupled blocks for grouped convergence."""
        basic_hierarchy = self._get_basic_coupled_hierarchy(hierarchy)
        basic_hierarchy_new = copy.deepcopy(basic_hierarchy)
        # First determine the partition IDs of the different functions and converger groups.
        for idx, entry in enumerate(basic_hierarchy):
            sublevel_list = entry[entry.keys()[0]]
            partitions_ids = [None]*len(sublevel_list)
            # Find out to which partition the converger dictionaries and separate functions belong
            for jdx, item in enumerate(sublevel_list):
                if isinstance(item, dict):
                    funcs = item[item.keys()[0]]
                    funcs_partition_ids = [None]*len(funcs)
                    for kdx, fun in enumerate(funcs):
                        for key, part_set in self.partition_sets.items():
                            if fun in part_set:
                                funcs_partition_ids[kdx] = key
                                continue
                    # Check that all partition IDs are the same
                    if not funcs_partition_ids.count(funcs_partition_ids[0]) == len(funcs_partition_ids):
                        raise AssertionError('The functions inside the converger {} do not belong to the same '
                                             'partition.'.format(item.keys()[0]))
                    else:
                        partitions_ids[jdx] = funcs_partition_ids[0]
                elif isinstance(item, str):
                    for key, part_set in self.partition_sets.items():
                        if item in part_set:
                            partitions_ids[jdx] = key
                            continue
            # If multiple items belong to multiple partitions, then create a partition group
            for part_id in self.partition_sets.keys():
                part_id_idxs = [i for i, j in enumerate(partitions_ids) if j == part_id]
                if len(part_id_idxs) > 1:  # create new partition group for this case
                    # Sort the part_id_idxs based on the step number
                    process_steps = []
                    updated_bhn = basic_hierarchy_new[idx][basic_hierarchy_new[idx].keys()[0]]
                    for part_id_idx in part_id_idxs:
                        list_item = updated_bhn[part_id_idx]
                        if isinstance(list_item, dict):
                            item_uid = list_item.keys()[0]
                        elif isinstance(list_item, str):
                            item_uid = list_item
                        else:
                            raise AssertionError('Could not map list_item.')
                        process_steps.append(self.block_step_numbers[item_uid])
                        part_id_idxs_sorted = [x for _, x in sorted(zip(process_steps, part_id_idxs))]
                    # Create a partition group and delete items that have become part of the group
                    part_dict = {'_Partition_{}'.format(part_id): []}
                    for part_id_idx in part_id_idxs_sorted:
                        part_dict['_Partition_{}'.format(part_id)].append(updated_bhn[part_id_idx])
                    for part_id_idx in sorted(part_id_idxs, reverse=True):
                        del basic_hierarchy_new[idx][entry.keys()[0]][part_id_idx]
                        del partitions_ids[part_id_idx]
                    # Add the new partition group to the hierarchy
                    basic_hierarchy_new[idx][entry.keys()[0]].append(part_dict)
        return basic_hierarchy_new

    def _get_basic_coupled_hierarchy(self, hierarchy):
        # type: (List) -> List[dict]
        """:obj:`list`: List containing the basic (no partitions) hierarchy of the coupled functions which defines the
        hierarchy of converged groups."""
        _coupled_hierarchy = []
        for entry in hierarchy:
            if isinstance(entry, dict):
                keys = entry.keys()
                if len(keys) != 1:
                    raise AssertionError('One key is expected in the dictionary of a process hierarchy.')
                if self.loop_element_types[keys[0]] == 'converger':
                    _coupled_hierarchy.append(entry)
                else:
                    return self._get_basic_coupled_hierarchy(entry[keys[0]])
        return _coupled_hierarchy

    @cached_property
    def model_sub_drivers(self):
        # type: () -> List[str]
        """:obj:`List[str]`: List with the subdrivers of the current model."""
        return [name for name in self.sub_drivers if self.SUBDRIVER_PREFIX + name in self.model_exec_blocks]

    @cached_property
    def model_super_drivers(self):
        # type: () -> List[str]
        """:obj:`List[str]`: List with the superdrivers of the current model."""
        return [name for name in self.super_drivers if self.SUPERDRIVER_PREFIX + name in self.all_loop_elements]

    @cached_property
    def model_super_components(self):
        # type: () -> List[str]
        """:obj:`List[str]`: List with the supercomponents of the current model."""
        return [name for name in self.model_exec_blocks if self.SUPERCOMP_PREFIX in name]

    @cached_property
    def system_inputs(self):
        # type: () -> Dict[str, int]
        """:obj:`Dict[str, int]`: Dictionary containing the system input sizes by their names."""
        _system_inputs = {}
        for value in self.elem_cmdows.xpath(
                    r'workflow/dataGraph/edges/edge[fromExecutableBlockUID="Coordinator"]/toParameterUID/text()'):
            if 'architectureNodes' not in value or 'designVariables' in value:
                if value in self.model_required_inputs or self.driver_uid in self.super_drivers:
                    name = xpath_to_param(value)
                    _system_inputs.update({name: self.variable_sizes[name]})
        return _system_inputs

    @cached_property
    def model_required_inputs(self):
        # type: () -> List[str]
        """:obj:List[str]`: List with all inputs that are required in the model."""
        _model_required_inputs = []
        for ex_block in self.model_exec_blocks:
            for value in self.elem_cmdows.xpath(r'workflow/dataGraph/edges/edge[toExecutableBlockUID="{}"]/'
                                                r'fromParameterUID/text()'.format(ex_block)):
                _model_required_inputs.append(value)
        return _model_required_inputs

    @cached_property
    def model_all_outputs(self):
        # type: () -> List[str]
        """:obj:List[str]`: List with all outputs that are provided in the model."""
        _model_all_outputs = []
        for ex_block in self.model_exec_blocks:
            for value in self.elem_cmdows.xpath(r'workflow/dataGraph/edges/edge[fromExecutableBlockUID="{}"]/'
                                                r'toParameterUID/text()'.format(ex_block)):
                _model_all_outputs.append(value)
        return _model_all_outputs

    @cached_property
    def model_super_inputs(self):
        # type: () -> Dict[str, dict]
        """:obj:`Dict[str, dict]`: Dictionary with the super inputs (keys) and their value and targets (values)."""
        _model_super_inputs = {}
        for super_driver in self.super_drivers:
            for value in self.elem_cmdows.xpath(r'workflow/dataGraph/edges/edge[fromExecutableBlockUID="{}"]/'
                                                r'toParameterUID/text()'.format(super_driver)):
                if value in self.model_required_inputs:
                    name = xpath_to_param(value)
                    if name not in _model_super_inputs:
                        # Determine the targets of this input
                        targets = [x for x in self.elem_cmdows.xpath(r'workflow/dataGraph/edges/edge[fromParameterUID'
                                                                     r'="{}"]/toExecutableBlockUID/text()'
                                                                     .format(value)) if x in self.model_exec_blocks]
                        _model_super_inputs.update({name: {'val': self.variable_sizes[name], 'targets': targets}})
        return _model_super_inputs

    @cached_property
    def model_super_inputs_inv(self):
        # type: () -> Dict[str]
        """:obj:`Dict[str]`: Inverse mapping of the model super inputs."""
        _model_super_inputs_inv_map = dict()
        for msi in self.model_super_inputs:
            if msi in self.mapped_parameters:
                if self.mapped_parameters[msi] not in _model_super_inputs_inv_map:
                    _model_super_inputs_inv_map.update({self.mapped_parameters[msi]: msi})
                else:
                    raise AssertionError('Model super input "{}" has already been mapped, cannot be mapped again.'
                                         .format(msi))
        return _model_super_inputs_inv_map

    @cached_property
    def model_constants(self):
        # type: () -> Dict[str]
        """:obj:`Dict[str]`: Constants used in the model."""
        _model_constants = {}
        for name, shape in self.system_inputs.items():
            if name not in self.design_vars.keys():
                _model_constants.update({name: shape})
        return _model_constants

    @cached_property
    def model_super_outputs(self):
        # type () -> Dict[str]
        """:obj:`Dict[str]`: Super outputs of the model to be used outside its own group."""
        _model_super_outputs = {}
        for output in self.model_all_outputs:
            for ex_block in self.elem_cmdows.xpath(r'workflow/dataGraph/edges/edge[fromParameterUID="{}"]/'
                                                   r'toExecutableBlockUID/text()'.format(output)):
                if ex_block not in self.model_exec_blocks and ex_block not in self.all_loop_elements:
                    if output in self.mapped_parameters:
                        output = self.mapped_parameters[output]
                    name = xpath_to_param(output)
                    _model_super_outputs.update({name: self.variable_sizes[name]})
                    continue
        return _model_super_outputs

    @cached_property
    def design_vars(self):
        # type: () -> Dict[str, Dict[str, Any]]
        """:obj:`dict`: Dictionary containing the design variables' initial values, lower bounds, and upper bounds."""
        if self.has_driver:
            desvars_uids = [elem.text for elem in
                            self.elem_model_driver.findall('designVariables/designVariable/designVariableUID')]
        else:
            desvars_uids = []
        if not desvars_uids:
            if self.has_driver:
                raise Exception('CMDOWS file {} does contain an optimizer, but no (valid) design variables'
                                .format(self.cmdows_path))
            else:
                return {}
        design_vars = {}
        for desvar_uid in desvars_uids:
            elem_desvar = get_element_by_uid(self.elem_cmdows, desvar_uid)
            name = xpath_to_param(elem_desvar.find('parameterUID').text)

            # Obtain the lower and upper bounds
            bounds = 2 * [None]  # type: List[Optional[float, np.array]]
            limit_range = elem_desvar.find('validRanges/limitRange')
            if limit_range is not None:
                for index, bnd, in enumerate(['minimum', 'maximum']):
                    elem = limit_range.find(bnd)
                    if elem is not None:
                        bounds[index] = parse_cmdows_value(elem)
                        if not self.does_value_fit(name, bounds[index]):
                            raise ValueError('incompatible size of {} for design variable {}'.format(bnd, name))
            else:
                bounds[0], bounds[1] = None, None

            # Obtain the initial value
            initial = elem_desvar.find('nominalValue')
            if initial is not None:
                initial = parse_cmdows_value(initial)
                if not self.does_value_fit(name, initial):
                    raise ValueError('Incompatible size of nominalValue for design variable "{}"'.format(name))
            else:
                if bounds[0] is None and bounds[1] is None:
                    warnings.warn('No nominalValue given for designVariable "{}". Default is all zeros.'.format(name))
                    initial = np.zeros(self.variable_sizes[name])
                else:
                    initial = (bounds[1] + bounds[0]) / 2.
                    warnings.warn('No nominalValue given for designVariable "{}". Defaulted to middle value w.r.t. '
                                  'bounds: {}.'.format(name, initial))

            # Add the design variable to the dict
            node_name = name if name not in self.coupling_vars else self.coupling_vars[name]['copy']

            # Check if the main driver is not a DOE with a Custom design table (then ref0 and ref should be None)
            ref0, ref = bounds[0], bounds[1]
            if self.has_doe:
                if self.elem_arch_elems.findtext('executableBlocks/does/doe/settings/method') == 'Custom design table':
                    ref0, ref = None, None
            design_vars.update({node_name: {'initial': initial,
                                            'lower': bounds[0], 'upper': bounds[1],
                                            'ref0': ref0, 'ref': ref}})
        return design_vars

    @cached_property
    def constraints(self):
        # type: () -> Dict[str, Dict[str, Any]]
        """:obj:`dict`: Dictionary containing the constraints' lower, upper, and equals reference values."""
        if self.elem_model_driver is not None:
            convars_uids = [elem.text for elem in
                            self.elem_model_driver.findall('constraintVariables/constraintVariable/constraintVariableUID')]
        else:
            convars_uids = []
        constraints = {}
        for convar_uid in convars_uids:
            elem_convar = get_element_by_uid(self.elem_cmdows, convar_uid)
            con = {'lower': None, 'upper': None, 'equals': None}
            param_uid = elem_convar.find('parameterUID').text
            if param_uid in self.mapped_parameters and 'architectureNodes/consistencyConstraint' not in param_uid:
                param_uid = self.mapped_parameters[param_uid]
            name = xpath_to_param(param_uid)

            if self.coupling_var_cons is not None and name in self.coupling_var_cons.values():
                # If this is a coupling variable consistency constraint, equals should just be zero
                for key, value in self.coupling_var_cons.items():
                    if name == value:
                        size = self.variable_sizes[key]
                        if size == 1:
                            con['equals'] = 0.
                        else:
                            con['equals'] = np.zeros(self.variable_sizes[key])
                        break
            else:
                # Obtain the reference value of the constraint
                constr_ref = elem_convar.find('referenceValue')  # type: etree._Element
                ref_vals = []
                ref = None
                if constr_ref is not None:
                    refs_str = constr_ref.text
                    if ';' in refs_str:
                        refs = refs_str.split(';')
                    else:
                        refs = [refs_str]
                    for ref in refs:
                        ref_val = parse_string(ref)
                        if isinstance(ref_val, str):
                            raise ValueError('referenceValue for constraint "%s" is not numerical' % name)
                        elif not self.does_value_fit(name, ref_val):
                            warnings.warn(
                                'incompatible size of constraint "%s". Will assume the same for all.' % name)
                            ref_val = np.ones(self.variable_sizes[name]) * np.atleast_1d(ref_val)[0]
                        ref_vals.append(ref_val)
                else:
                    warnings.warn('no referenceValue given for constraint "%s". Default is all zeros.' % name)
                    ref_vals = [np.zeros(self.variable_sizes[name])]

                # Process the constraint type
                constr_type = elem_convar.find('constraintType')
                if constr_type is not None:
                    if constr_type.text == 'inequality':
                        constr_oper = elem_convar.find('constraintOperator')
                        if constr_oper is not None:
                            opers_str = constr_oper.text
                            if ';' in opers_str:
                                opers = opers_str.split(';')
                            else:
                                opers = [opers_str]
                            for idx, oper in enumerate(opers):
                                if oper == '>=' or oper == '>':
                                    con['lower'] = ref_vals[idx]
                                elif oper == '<=' or oper == '<':
                                    con['upper'] = ref_vals[idx]
                                else:
                                    raise ValueError(
                                        'invalid constraintOperator "%s" for constraint "%s"' % (oper, name))
                        else:
                            warnings.warn(
                                'no constraintOperator given for inequality constraint. Default is "&lt;=".')
                            con['upper'] = ref_vals[0]
                    elif constr_type.text == 'equality':
                        if elem_convar.find('constraintOperator') is not None:
                            if elem_convar.find('constraintOperator').text != '==':
                                warnings.warn('constraintOperator given for an equalityConstraint will be ignored')
                        con['equals'] = ref_vals[0]
                    else:
                        raise ValueError(
                            'invalid constraintType "%s" for constraint "%s".' % (constr_type.text, name))
                else:
                    warnings.warn('no constraintType specified for constraint "%s". Default is a <= inequality.')
                    con['upper'] = ref

            # Add constraint to the dictionary
            constraints.update({name: con})
        return constraints

    @cached_property
    def objective(self):
        # type: () -> str
        """:obj:`str`: Name of the objective variable."""
        if self.objective_required:
            uid_obj = self.elem_model_driver.findtext('objectiveVariables/objectiveVariable/objectiveVariableUID')
            if uid_obj is None:
                raise InvalidCMDOWSFileError('does not contain (valid) objective variable')
            obj_elem = get_element_by_uid(self.elem_cmdows, uid_obj)
            return xpath_to_param(obj_elem.findtext('parameterUID'))
        else:
            pass

    def _configure_coupled_groups(self, hierarchy, root=True):
        # type: (List) -> Optional[Group]
        """:obj:`Group`, optional: Group wrapping the coupled blocks with a converger specified in the CMDOWS file.

        This method enables the iterative configuration of groups of distributed convergers based on the convergence
        hierarchy.
        """
        subsys = None
        if not root:
            coupled_group = Group()

        for entry in hierarchy:
            if isinstance(entry, dict):  # if entry specifies a coupled group
                uid = entry.keys()[0]
                if root:
                    subsys = self.add_subsystem(str_to_valid_sys_name(uid),
                                                self._configure_coupled_groups(entry[uid], False), ['*'])
                else:
                    subsys = coupled_group.add_subsystem(str_to_valid_sys_name(uid),
                                                         self._configure_coupled_groups(entry[uid], False), ['*'])
                if '_Partition_' not in uid:
                    conv_elem = get_element_by_uid(self.elem_arch_elems, uid)
                else:
                    conv_elem = None
                # Define linear solver
                if conv_elem is not None:
                    linsol_elem = conv_elem.find('settings/linearSolver')
                    if isinstance(linsol_elem, _Element):
                        if linsol_elem.find('method').text == 'Gauss-Seidel':
                            linsol = subsys.linear_solver = LinearBlockGS()
                        elif linsol_elem.find('method').text == 'Jacobi':
                            linsol = subsys.linear_solver = LinearBlockJac()
                        else:
                            raise ValueError('Specified convergerType "{}" is not supported.'
                                             .format(linsol_elem.find('method').text))
                        linsol.options['maxiter'] = int(linsol_elem.find('maximumIterations').text)
                        linsol.options['atol'] = float(linsol_elem.find('convergenceToleranceAbsolute').text)
                        linsol.options['rtol'] = float(linsol_elem.find('convergenceToleranceRelative').text)
                    else:
                        subsys.linear_solver = DirectSolver()
                        warnings.warn('Linear solver was not defined in CMDOWS file for converger {}. linear_solver set'
                                      ' to default "DirectSolver()".'.format(str_to_valid_sys_name(uid)))
                else:
                    subsys.linear_solver = LinearRunOnce()

                # Define nonlinear solver
                if conv_elem is not None:
                    nonlinsol_elem = conv_elem.find('settings/nonlinearSolver')
                    if isinstance(nonlinsol_elem, _Element):
                        if nonlinsol_elem.find('method').text == 'Gauss-Seidel':
                            nonlinsol = subsys.nonlinear_solver = NonlinearBlockGS()
                        elif nonlinsol_elem.find('method').text == 'Jacobi':
                            nonlinsol = subsys.nonlinear_solver = NonlinearBlockJac()
                        else:
                            raise ValueError('Specified convergerType "{}" is not supported.'
                                             .format(nonlinsol_elem.find('method').text))
                        nonlinsol.options['maxiter'] = int(nonlinsol_elem.find('maximumIterations').text)
                        nonlinsol.options['atol'] = float(nonlinsol_elem.find('convergenceToleranceAbsolute').text)
                        nonlinsol.options['rtol'] = float(nonlinsol_elem.find('convergenceToleranceRelative').text)
                    else:
                        subsys.nonlinear_solver = NonlinearRunOnce()
                        warnings.warn('Nonlinear solver was not defined in CMDOWS file for converger {}. '
                                      'nonlinear_solver set to default "NonlinearRunOnce()".'
                                      .format(str_to_valid_sys_name(uid)))
                else:
                    subsys.nonlinear_solver = NonlinearRunOnce()
            elif isinstance(entry, str):  # if entry specifies an executable block
                if root:
                    raise AssertionError('Code was not expected to get here for root == True.')
                # Get the correct DisciplineComponent or MathematicalFunction
                promotes = ['*']  # type: List[Union[str, Tuple[str, str]]]
                uid = entry
                if uid in self.discipline_components:
                    block = self.discipline_components[uid]
                elif uid in self.mathematical_functions_groups:
                    block = self.mathematical_functions_groups[uid]
                else:
                    raise RuntimeError('uID {} not found in discipline_components, nor mathematical_functions_groups.'
                                       .format(uid))
                # Add the block to the group
                coupled_group.add_subsystem(str_to_valid_sys_name(uid), block, promotes)
            else:
                raise ValueError('Unexpected value type {} encountered in the coupled_hierarchy {}.'
                                 .format(type(entry), hierarchy))
        if root:
            return subsys
        else:
            return coupled_group

    @cached_property
    def system_order(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: List system names in the order specified in the CMDOWS file."""
        _system_order = ['coordinator']
        for name in self.model_super_drivers:
            _system_order.append(str_to_valid_sys_name(name))

        coupled_group_set = False
        n = 0
        for block in self.block_order:
            if block in self.coupled_blocks:
                n += 1
                if not coupled_group_set:
                    for entry in self.coupled_hierarchy:
                        _system_order.append(str_to_valid_sys_name(entry.keys()[0]))
                    coupled_group_set = True
            elif block in self.model_exec_blocks or self.SUBDRIVER_PREFIX + block in self.model_exec_blocks:
                n += 1
                _system_order.append(str_to_valid_sys_name(block))

        if len(self.discipline_components) + len(self.mathematical_functions_groups) < n:
            raise InvalidCMDOWSFileError('something is wrong with the executableBlocksOrder')

        return _system_order

    @cached_property
    def coordinator(self):
        # type: () -> IndepVarComp
        """:obj:`IndepVarComp`: An `IndepVarComp` representing the system's ``Coordinator`` block.

        This `IndepVarComp` takes care of all system input parameters and initial values of design variables.
        """
        coordinator = IndepVarComp()

        # Add design variables
        for name, value in self.design_vars.items():
            coordinator.add_output(name, value['initial'])

        # Add system constants
        for name, shape in self.model_constants.items():
            coordinator.add_output(name, shape=shape)

        return coordinator

    def configure_super_driver(self, name):
        # type: () -> IndepVarComp
        """:obj:`IndepVarComp`: An `IndepVarComp` representing a super driver in a subdriver system."""
        super_driver = IndepVarComp()

        # Add superdriver outputs
        for value in self.elem_cmdows.xpath(
                r'workflow/dataGraph/edges/edge[fromExecutableBlockUID="{}"]/toParameterUID/text()'.format(name)):
            if 'architectureNodes/finalDesignVariables' not in value and value in self.model_required_inputs:
                name = xpath_to_param(value)
                super_driver.add_output(xpath_to_param(value), self.variable_sizes[name])
        return super_driver

    def setup(self):
        # type: () -> None
        """Assemble the LEGOModel using the the CMDOWS file and knowledge base."""
        # Add the coordinator
        self.add_subsystem('coordinator', self.coordinator, ['*'])

        # Add superdrivers as IndepVarComps
        for name in self.model_super_drivers:
            self.add_subsystem(str_to_valid_sys_name(name), self.configure_super_driver(name), ['*'])

        # Add all pre-coupling and post-coupling components
        for name, component in self.discipline_components.items():
            if name not in self.coupled_blocks and name in self.model_exec_blocks:
                promotes = ['*']
                # Change input variable names if they are provided as copies of coupling variables
                for i in component.inputs_from_xml.keys():
                    if i in self.coupling_vars:
                        if isinstance(self.coupling_vars[i], dict):
                            if 'copy' in self.coupling_vars[i]:
                                promotes.append((i, self.coupling_vars[i]['copy']))
                    elif i in self.des_var_copies:
                        promotes.append((i, self.des_var_copies[i]))
                    elif i in self.model_super_inputs_inv:
                        mapped_var = self.model_super_inputs_inv[i]
                        if mapped_var in self.model_super_inputs:
                            if name in self.model_super_inputs[mapped_var]['targets']:
                                promotes.append((i, mapped_var))
                self.add_subsystem(str_to_valid_sys_name(name), component, promotes)
        for name, component in self.mathematical_functions_groups.items():
            if name not in self.coupled_blocks and name in self.model_exec_blocks:
                self.add_subsystem(str_to_valid_sys_name(name), component, ['*'])

        # Add the coupled groups
        if self.coupled_hierarchy:
            self._configure_coupled_groups(self.coupled_hierarchy, True)

        # Add the subdriver groups
        for name in self.model_sub_drivers:
            from openlego.core.subdriver_component import SubDriverComponent
            split_file_name = os.path.splitext(self.base_xml_file)
            base_xml_file = split_file_name[0] + '_' + name + split_file_name[1]
            self.add_subsystem(str_to_valid_sys_name(name),
                               SubDriverComponent(cmdows_path=self.cmdows_path,
                                                  driver_uid=name,
                                                  kb_path=self.kb_path,
                                                  data_folder=self.data_folder,
                                                  base_xml_file=base_xml_file,
                                                  show_model=False), promotes=['*'])

        # Put the blocks in the correct order
        self.set_order(list(self.system_order))

        # Add the design variables
        for name, value in self.design_vars.items():
            self.add_design_var(name, lower=value['lower'], upper=value['upper'], ref0=value['ref0'], ref=value['ref'])

        # Add the constraints
        for name, value in self.constraints.items():
            self.add_constraint(name, lower=value['lower'], upper=value['upper'], equals=value['equals'])

        # Add the objective
        if self.objective:
            self.add_objective(self.objective)

    def initialize_from_xml(self, xml):
        # type: (Union[str, _ElementTree]) -> None
        """Initialize the problem with initial values from an XML file.

        This function can only be called after the problem's setup method has been called.

        Parameters
        ----------
            xml : str or :obj:`etree._ElementTree`
                Path to an XML file or an instance of `etree._ElementTree` representing it.
        """
        for xpath, value in xml_to_dict(xml).items():
            name = xpath_to_param(xpath)
            if name in self._outputs:
                self._outputs[name] = value
            elif name in self._inputs:
                self._inputs[name] = value
            if name in self.mapped_parameters_inv:
                for mapping in self.mapped_parameters_inv[name]:
                    if mapping in self._outputs:
                        self._outputs[mapping] = value
                    try:
                        if mapping in self._inputs:
                            self._inputs[mapping] = value
                    except RuntimeError as e:
                        if 'The promoted name' in e[0] and 'is invalid' in e[0]:
                            warnings.warn('Could not automatically set this invalid promoted name from the XML: {}.'
                                          .format(mapping))
                        else:
                            raise RuntimeError(e)
