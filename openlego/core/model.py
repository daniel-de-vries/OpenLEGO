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

This file contains the definition of the `LEGOModel` class.
"""
from __future__ import absolute_import, division, print_function

import imp
import os
import warnings

import numpy as np
from lxml import etree
from lxml.etree import _Element, _ElementTree
from openmdao.api import Group, IndepVarComp, LinearBlockGS, NonlinearBlockGS, LinearBlockJac, NonlinearBlockJac, \
    LinearRunOnce, ExecComp, NonlinearRunOnce, DirectSolver
from typing import Union, Optional, List, Any, Dict, Tuple

from openlego.utils.general_utils import CachedProperty, parse_cmdows_value, str_to_valid_sys_name, parse_string
from openlego.utils.xml_utils import xpath_to_param, xml_to_dict
from openlego.utils.cmdows_utils import get_element_by_uid, get_related_parameter_uid, get_loop_nesting_obj
from .abstract_discipline import AbstractDiscipline
from .cmdows_object import CMDOWSObject
from .discipline_component import DisciplineComponent


class InvalidCMDOWSFileError(ValueError):

    def __init__(self, reason=None):
        msg = 'Invalid CMDOWS file'
        if reason is not None:
            msg += ': {}'.format(reason)
        super(InvalidCMDOWSFileError, self).__init__(msg)


class LEGOModel(CMDOWSObject, Group):
    """Specialized OpenMDAO Group class representing the model specified by a CMDOWS file.

    An important note about this class in the context of OpenMDAO is that the aggregation pattern of the root Group
    class the base Problem class has is changed into a stronger composition pattern. This is because this class directly
    controls the creation and assembly of this class by making use of Python's @property decorator. It is not possible,
    nor should it be attempted, to manually inject a different instance of Group  in place of these, because the
    correspondence between the CMDOWS file and the Problem can then no longer be guaranteed.

    Attributes
    ----------
        cmdows_path
        kb_path
        discipline_components
        block_order
        coupled_blocks
        system_order
        system_variables
        system_inputs
        driver
        coordinator

        data_folder : str, optional
            Path to the folder in which to store all data generated during the `Problem`'s execution.

        base_xml_file : str, optional
            Path to an XML file which should be kept up-to-date with the latest data describing the problem.
    """

    def __init__(self, cmdows_path=None, kb_path='', data_folder=None, base_xml_file=None, **kwargs):
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
        super(LEGOModel, self).__init__(cmdows_path, kb_path, data_folder, base_xml_file, **kwargs)

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

    @CachedProperty
    def elem_cmdows(self):
        # type: () -> _Element
        """:obj:`etree._Element`: Root element of the CMDOWS XML file."""
        return etree.parse(self.cmdows_path).getroot()

    @CachedProperty
    def elem_problem_def(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The problemDefition element of the CMDOWS file."""
        return self.elem_cmdows.find('problemDefinition')

    @CachedProperty
    def elem_workflow(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The workflow element of the CMDOWS file."""
        return self.elem_cmdows.find('workflow')

    @CachedProperty
    def elem_params(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The problemRoles/parameters element of the CMDOWS file."""
        params = self.elem_cmdows.find('problemDefinition/problemRoles/parameters')
        if params is None:
            raise InvalidCMDOWSFileError('does not contain (valid) parameters in the problemRoles')
        return params

    @CachedProperty
    def elem_arch_elems(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The architectureElements element of the CMDOWS file."""
        arch_elems = self.elem_cmdows.find('architectureElements')
        if arch_elems is None:
            raise InvalidCMDOWSFileError('does not contain (valid) architecture elements')
        return arch_elems

    @CachedProperty
    def elem_loop_nesting(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The loopNesting element of the CMDOWS file."""
        loopnesting_elem = self.elem_workflow.find('processGraph/metadata/loopNesting')
        if loopnesting_elem is None:
            raise InvalidCMDOWSFileError('does not contain loopNesting element')
        return loopnesting_elem

    @CachedProperty
    def has_optimizer(self):
        # type: () -> bool
        """:obj:`bool`: True if there is an optimizer, False if not."""
        if self.elem_arch_elems.find('executableBlocks/optimizers/optimizer') is not None:
            return True
        return False

    @CachedProperty
    def has_doe(self):
        # type: () -> bool
        """:obj:`bool`: True if there is a DOE component, False if not."""
        if self.elem_arch_elems.find('executableBlocks/does/doe') is not None:
            return True
        return False

    @CachedProperty
    def has_driver(self):
        # type: () -> bool
        """:obj:`bool`: True if there is a driver component (DOE or optimizer), False if not."""
        if self.has_doe or self.has_optimizer:
            return True
        return False

    @CachedProperty
    def has_converger(self):
        # type: () -> bool
        """:obj:`bool`: True if there is a converger, False if not."""
        if self.elem_arch_elems.find('executableBlocks/convergers/converger') is not None:
            return True
        return False

    @CachedProperty
    def objective_required(self):
        # type: () -> bool
        """:obj:`bool`: True if an objective value is required, False if not."""
        if self.has_optimizer:
            return True
        return False

    @CachedProperty
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

    @CachedProperty
    def mapped_parameters(self):
        # type: () -> Dict[str, str]
        """:obj:`dict`: Dictionary of parameters that are mapped in the CMDOWS file, for example as copies."""
        mapped_params = dict()
        for elem_category in self.elem_arch_elems.find('parameters').iterchildren():
            for elem_param in elem_category.iterchildren():
                param, mapped = get_related_parameter_uid(elem_param, self.elem_cmdows)
                mapped_params.update({param: mapped})
        return mapped_params

    @CachedProperty
    def mathematical_functions_inputs(self):
        # type: () -> Dict[str, List[Tuple[str]]]
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

    @CachedProperty
    def mathematical_functions_outputs(self):
        # type: () -> Dict[str, List[Tuple[str]]]
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

    @CachedProperty
    def mathematical_functions_groups(self):
        # type: () -> Dict[str, Group]
        """:obj:`dict`: Dictionary of execute components by their mathematical function ``uID`` from CMDOWS.
        """
        _mathematical_functions = dict()
        for mathematical_function in self.elem_cmdows.iter('mathematicalFunction'):
            uid = mathematical_function.attrib['uID']
            group = Group()

            eq_mapping = dict()
            for output in mathematical_function.iter('output'):
                if output.find('equations') is not None:
                    for equation in output.iter('equation'):
                        if equation.attrib['language'] == 'Python':
                            eq_uid = equation.getparent().attrib['uID']
                            eq_expr = equation.text
                            eq_output = xpath_to_param(output.find('parameterUID').text)

                            promotes = list()
                            for eq_label, input_name in self.mathematical_functions_inputs[uid]:
                                # TODO: This mapping of the input name should get a more sophisticated logic
                                if input_name in self.mapped_parameters and input_name not in self.design_vars:
                                    input_name = self.mapped_parameters[input_name]

                                if eq_label in eq_expr:
                                    promotes.append((eq_label, input_name))

                            eq_mapping.update({eq_uid: eq_output})
                            group.add_subsystem(str_to_valid_sys_name(eq_uid),
                                                ExecComp('output = ' + eq_expr),
                                                promotes=promotes + [('output', eq_output), ])

            extra_output_counter = 0
            for output in mathematical_function.iter('output'):
                eq_uid = output.find('equationsUID')
                if eq_uid is not None:
                    eq_uid = eq_uid.text
                    eq_output = xpath_to_param(output.find('parameterUID').text)

                    group.add_subsystem('extra_output_{}'.format(extra_output_counter),
                                        ExecComp('output = input'),
                                        promotes=[('output', eq_output), ('input', eq_mapping[eq_uid])])
                    extra_output_counter += 1

            _mathematical_functions.update({uid: group})

        return _mathematical_functions

    @CachedProperty
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

    @CachedProperty
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

    @CachedProperty
    def coupling_var_copies(self):
        # type: () -> Dict[str, str]
        """:obj:`dict`: Dictionary with coupling variable copies."""
        coupling_var_copies = dict()
        for var, value in self.coupling_vars.items():
            coupling_var_copies.update({var: value['copy']})
        return coupling_var_copies

    @CachedProperty
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

    @CachedProperty
    def block_order(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: List of executable block ``uIDs`` in the order specified in the CMDOWS file."""
        positions = list()
        uids = list()
        for block in self.elem_workflow.iterfind('processGraph/metadata/executableBlocksOrder/executableBlock'):
            uid = block.text
            positions.append(int(block.attrib['position']))
            uids.append(uid)
        return [uid for position, uid in sorted(zip(positions, uids))]

    @CachedProperty
    def coupled_blocks(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: List of ``uIDs`` of the coupled executable blocks specified in the CMDOWS file."""
        _coupled_blocks = []
        for block in self.elem_arch_elems.iterfind(
                'executableBlocks/coupledAnalyses/coupledAnalysis/relatedExecutableBlockUID'):
            _coupled_blocks.append(block.text)
        return _coupled_blocks

    @CachedProperty
    def loop_nesting_dict(self):
        # type: () -> Dict[str, dict]
        """:obj:`dict`: Dictionary of the loopNesting XML element."""
        return get_loop_nesting_obj(self.elem_loop_nesting)

    @CachedProperty
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

    @CachedProperty
    def coupled_hierarchy(self):
        # type: () -> List[dict]
        """:obj:`list`: List containing the hierarchy of the coupled blocks for grouped convergence."""
        return self._get_coupled_hierarchy(self.loop_nesting_dict)

    def _get_coupled_hierarchy(self, hierarchy):
        # type: (List) -> List[dict]
        """:obj:`list`: List containing the hierarchy of the coupled functions which defines the hierarchy of converged
        groups."""
        _coupled_hierarchy = []
        for entry in hierarchy:
            if isinstance(entry, dict):
                keys = entry.keys()
                if len(keys) != 1:
                    raise AssertionError('One key is expected in the dictionary of a process hierarchy.')
                if self.loop_element_details[keys[0]] == 'converger':
                    _coupled_hierarchy.append(entry)
                else:
                    return self._get_coupled_hierarchy(entry[keys[0]])
        return _coupled_hierarchy

    @CachedProperty
    def system_inputs(self):
        # type: () -> Dict[str, int]
        """:obj:`dict`: Dictionary containing the system input sizes by their names."""
        system_inputs = {}
        for value in self.elem_cmdows.xpath(
                    r'workflow/dataGraph/edges/edge[fromExecutableBlockUID="Coordinator"]/toParameterUID/text()'):
            if 'architectureNodes' not in value or 'designVariables' in value:
                name = xpath_to_param(value)
                system_inputs.update({name: self.variable_sizes[name]})

        return system_inputs

    @CachedProperty
    def design_vars(self):
        # type: () -> Dict[str, Dict[str, Any]]
        """:obj:`dict`: Dictionary containing the design variables' initial values, lower bounds, and upper bounds."""
        desvars = self.elem_params.find('designVariables')
        if desvars is None:
            if self.has_driver:
                raise Exception(
                    'CMDOWS file {} does contain an optimizer, but no (valid) design variables'.format(self.cmdows_path)
                )
            else:
                return {}
        design_vars = {}
        for desvar in desvars:
            name = xpath_to_param(desvar.find('parameterUID').text)

            # Obtain the initial value
            initial = desvar.find('nominalValue')
            if initial is not None:
                initial = parse_cmdows_value(initial)
                if not self.does_value_fit(name, initial):
                    raise ValueError('incompatible size of nominalValue for design variable "%s"' % name)
            else:
                warnings.warn('no nominalValue given for designVariable "%s". Default is all zeros.' % name)
                initial = np.zeros(self.variable_sizes[name])

            # Obtain the lower and upper bounds
            bounds = 2 * [None]  # type: List[Optional[str]]
            limit_range = desvar.find('validRanges/limitRange')
            if limit_range is not None:
                for index, bnd, in enumerate(['minimum', 'maximum']):
                    elem = limit_range.find(bnd)
                    if elem is not None:
                        bounds[index] = parse_cmdows_value(elem)
                        if not self.does_value_fit(name, bounds[index]):
                            raise ValueError('incompatible size of %s for design variable %s' % (bnd, name))
            else:
                bounds[0] = None
                bounds[1] = None

            # Add the design variable to the dict
            node_name = name if name not in self.coupling_vars else self.coupling_vars[name]['copy']
            design_vars.update({node_name: {'initial': initial,
                                            'lower': bounds[0], 'upper': bounds[1],
                                            'ref0': bounds[0], 'ref': bounds[1]}})
        return design_vars

    @CachedProperty
    def constraints(self):
        # type: () -> Dict[str, Dict[str, Any]]
        """:obj:`dict`: Dictionary containing the constraints' lower, upper, and equals reference values."""
        convars = self.elem_params.find('constraintVariables')
        constraints = {}
        if convars is not None:
            for convar in convars:
                con = {'lower': None, 'upper': None, 'equals': None}
                name = xpath_to_param(convar.find('parameterUID').text)

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
                    constr_ref = convar.find('referenceValue')  # type: etree._Element
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
                    constr_type = convar.find('constraintType')
                    if constr_type is not None:
                        if constr_type.text == 'inequality':
                            constr_oper = convar.find('constraintOperator')
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
                            if convar.find('constraintOperator') is not None:
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

    @CachedProperty
    def objective(self):
        # type: () -> str
        """:obj:`str`: Name of the objective variable."""
        objvars = self.elem_params.find('objectiveVariables')
        if self.objective_required:
            if objvars is None:
                raise InvalidCMDOWSFileError('does not contain (valid) objective variables')
            if len(objvars) > 1:
                raise InvalidCMDOWSFileError('contains multiple objectives, but this is not supported')
            return xpath_to_param(objvars[0].find('parameterUID').text)
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
                conv_elem = get_element_by_uid(self.elem_arch_elems, uid)
                # Define linear solver
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
                    warnings.warn('Linear solver was not defined in CMDOWS file for converger {}. linear_solver set to'
                                  ' default "DirectSolver()".'.format(str_to_valid_sys_name(uid)))

                # Define nonlinear solver
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
                    warnings.warn('Nonlinear solver was not defined in CMDOWS file for converger {}. nonlinear_solver'
                                  ' set to default "NonlinearRunOnce()".'.format(str_to_valid_sys_name(uid)))
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

    # TODO: implement this property in phase 3 (required for distributed architectures such as BLISS-2000 and CO)
    @CachedProperty
    def subsystem_optimization_groups(self):
        # type: () -> Optional[list]
        """:obj:`list`, optional: list containing groups of suboptimizations used in distributed architectures.

        If not subsystem optimizations are required based on the CMDOWS file then this property is `None`.
        """
        if self.subsystem_optimizations:
            return None
            # Add inputs (standardized?)
            # Add output
            # Declare partials
            # Set subproblem
            # Define copies in params subsystem
            # Define design variables
            # Define components
            # Connect everything together (through promotion?)
            # Set subproblem optimizer
            # Add design variables, objective, constraints
            # Setup and final setup

    @CachedProperty
    def system_order(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: List system names in the order specified in the CMDOWS file."""
        _system_order = ['coordinator']
        coupled_group_set = False
        n = 0
        for block in self.block_order:
            if block in self.coupled_blocks:
                n += 1
                if not coupled_group_set:
                    for entry in self.coupled_hierarchy:
                        _system_order.append(str_to_valid_sys_name(entry.keys()[0]))
                    coupled_group_set = True
            elif block in self.discipline_components or block in self.mathematical_functions_groups:
                n += 1
                _system_order.append(str_to_valid_sys_name(block))

        if n < len(self.discipline_components) + len(self.mathematical_functions_groups):
            raise InvalidCMDOWSFileError('executableBlocksOrder is incomplete')

        return _system_order

    @CachedProperty
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
        for name, shape in self.system_inputs.items():
            if name not in self.design_vars.keys():
                coordinator.add_output(name, shape=shape)

        return coordinator

    def setup(self):
        # type: () -> None
        """Assemble the LEGOModel using the the CMDOWS file and knowledge base."""
        # Add the coordinator
        self.add_subsystem('coordinator', self.coordinator, ['*'])

        # Add all pre-coupling and post-coupling components
        for name, component in self.discipline_components.items():
            if name not in self.coupled_blocks:
                promotes = ['*']
                # Change input variable names if they are provided as copies of coupling variables
                for i in component.inputs_from_xml.keys():
                    if i in self.coupling_vars:
                        if isinstance(self.coupling_vars[i], dict):
                            if 'copy' in self.coupling_vars[i]:
                                promotes.append((i, self.coupling_vars[i]['copy']))
                self.add_subsystem(str_to_valid_sys_name(name), component, promotes)
        for name, component in self.mathematical_functions_groups.items():
            if name not in self.coupled_blocks:
                # TODO: Adjust promotion of variables for copies?
                self.add_subsystem(str_to_valid_sys_name(name), component, ['*'])

        # Add the coupled groups
        if self.coupled_hierarchy:
            self._configure_coupled_groups(self.coupled_hierarchy, True)

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
