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

This file contains the definition of the `CMDOWSProblem` class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import imp
import numpy as np
import warnings

from lxml import etree
from lxml.etree import _Element, _ElementTree
from openmdao.api import Problem, Group, LinearGaussSeidel, NLGaussSeidel, IndepVarComp, Driver, ExecComp
from typing import Union, Optional, List, Any, Dict

from openlego.discipline import AbstractDiscipline
from openlego.components import DisciplineComponent
from openlego.xml import xpath_to_param, xml_to_dict
from openlego.util import CachedProperty, parse_cmdows_value


class CMDOWSProblem(Problem):
    """Specialized OpenMDAO Problem class representing the problem specified by a CMDOWS file.

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

    def __init__(self, cmdows_path=None, kb_path=None, driver=None, data_folder=None, base_xml_file=None):
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

        driver : :obj:`Driver`, optional
            Instance of a Driver.

        data_folder : str, optional
            Path to the data folder in which to store all files and output from the problem.

        base_xml_file : str, optional
            Path to a base XML file to update with the problem data.
        """
        self._cmdows_path = cmdows_path
        self._kb_path = kb_path
        self.data_folder = data_folder
        self.base_xml_file = base_xml_file

        self.__coupled_group_promotes = None
        self.__coordinator_promotes = None

        self._driver = None

        super(CMDOWSProblem, self).__init__(driver=driver)

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
            if name in [_name for _name, value in self.__class__.__dict__.items() if isinstance(value, CachedProperty)]:
                self.__integrity_check()
        return super(CMDOWSProblem, self).__getattribute__(name)

    def __integrity_check(self):
        # type: () -> None
        """Ensure both a CMDOWS file and a knowledge base path have been supplied.

        Raises
        ------
            ValueError
                If either no CMDOWS file or no knowledge base path has been supplied
        """
        a = self._cmdows_path is None
        b = self._kb_path is None
        if a or b:
            raise ValueError('No ' + a * 'CMDOWS file ' + (a & b) * 'and ' + b * 'knowledge base path ' + 'specified!')

    def invalidate(self):
        # type: () -> None
        """Invalidate the instance.

        All computed (cached) properties will be recomputed upon being read once the instance has been invalidated."""
        for value in self.__class__.__dict__.values():
            if isinstance(value, CachedProperty):
                value.invalidate()

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

    @CachedProperty
    def elem_cmdows(self):
        # type: () -> _Element
        """:obj:`etree._Element`: Root element of the CMDOWS XML file."""
        return etree.parse(self.cmdows_path).getroot()

    @CachedProperty
    def elem_problem_def(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The problemDefition element of this problem's CMDOWS file."""
        return self.elem_cmdows.find('problemDefinition')

    @CachedProperty
    def elem_params(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The problemRoles/parameters element of the CMDOWS file."""
        params = self.elem_cmdows.find('problemDefinition/problemRoles/parameters')
        if params is None:
            raise Exception('cmdows does not contain (valid) parameters in the problemRoles')
        return params

    @CachedProperty
    def elem_arch_elems(self):
        # type: () -> _Element
        """:obj:`etree._Element`: The architectureElements element of the CMDOWS file."""
        arch_elems = self.elem_cmdows.find('architectureElements')
        if arch_elems is None:
            raise Exception('cmdows does not contain (valid) architecture elements')
        return arch_elems

    @CachedProperty
    def has_converger(self):
        # type: () -> bool
        """:obj:`bool`: True if there is a converger, False if not."""
        if self.elem_arch_elems.find('converger') is not None:
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
        _discipline_components = dict()
        for design_competence in self.elem_cmdows.iter('designCompetence'):
            uid = design_competence.attrib['uID']
            name = design_competence.find('ID').text
            try:
                fp, pathname, description = imp.find_module(name, [self.kb_path])
                mod = imp.load_module(name, fp, pathname, description)
                cls = getattr(mod, name)  # type: AbstractDiscipline.__class__
                if not issubclass(cls, AbstractDiscipline):
                    raise RuntimeError
            except:
                raise RuntimeError(
                    'Unable to process CMDOWS file: no proper discipline found for design competence with name %s'
                    % name)
            finally:
                if 'fp' in locals():
                    fp.close()

            component = DisciplineComponent(cls(), data_folder=self.data_folder, base_file=self.base_xml_file)
            _discipline_components.update({uid: component})
        return _discipline_components

    @CachedProperty
    def variable_sizes(self):
        # type: () -> Dict[str, int]
        """:obj:`dict`: Dictionary of the sizes of all variables by their names."""
        variable_sizes = {}
        for component in self.discipline_components.values():
            for name, value in component.variables_from_xml.items():
                variable_sizes.update({name: np.atleast_1d(value).size})
        return variable_sizes

    @CachedProperty
    def coupling_vars(self):
        # type: () -> Dict[str, Dict[str, str]]
        """:obj:`dict`: Dictionary with coupling variables."""
        coupling_vars = dict()

        # First create a map between related param and coupling copy var
        for var in self.elem_arch_elems.iter('couplingCopyVariable'):
            related_param = var.find('relatedParameterUID').text
            coupling_vars.update({xpath_to_param(related_param): xpath_to_param(var.attrib['uID'])})

        # Then update dict with corresponding consitency constraint var
        for convar in self.elem_arch_elems.iter('consistencyConstraintVariable'):
            param = xpath_to_param(convar.find('relatedParameterUID').text)
            if param not in coupling_vars:
                raise RuntimeError('invalid cmdows file')

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
        # type: () -> Dict[str, str]
        """:obj:`dict`: Dictionary with coupling variable constraints."""
        coupling_var_cons = dict()
        for var, value in self.coupling_vars.items():
            coupling_var_cons.update({var: value['con']})
        return coupling_var_cons

    @CachedProperty
    def block_order(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: List of executable block ``uIDs`` in the order specified in the CMDOWS file."""
        positions = list()
        uids = list()
        for block in self.elem_problem_def.iterfind('problemFormulation/executableBlocksOrder/executableBlock'):
            uid = block.text
            positions.append(int(block.attrib['position']))
            uids.append(uid)
        return [uid for position, uid in sorted(zip(positions, uids))]

    @CachedProperty
    def coupled_blocks(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: List of ``uIDs`` of the coupled executable blocks specified in the CMDOWS file."""
        _coupled_blocks = []
        for block in self.elem_problem_def.iterfind('problemRoles/executableBlocks/coupledBlocks/coupledBlock'):
            _coupled_blocks.append(block.text)
        return _coupled_blocks

    @CachedProperty
    def system_order(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: List system names in the order specified in the CMDOWS file."""
        _system_order = ['coordinator']
        coupled_group_set = False
        for block in self.block_order:
            if block in self.coupled_blocks:
                if not coupled_group_set:
                    _system_order.append('coupled_group')
                    coupled_group_set = True
            elif block in self.discipline_components:
                _system_order.append(block)
        if self.consistency_constraint_group is not None:
            _system_order.append('consistency_constraints')
        return _system_order

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
            raise Exception('cmdows does not contain (valid) design variables')

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

            if name in self.coupling_vars:
                # If this is a coupling variable the bounds are -1e99 and 1e99
                b = 1e99*np.ones(self.variable_sizes[name])
                bounds = [-b, b]

                # Change the name to the name stated for the copy of this coupling variable
                name = self.coupling_vars[name]['copy']
            else:
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

            # Add the design variable to the dict
            design_vars.update({name: {'initial': initial,
                                       'lower': bounds[0], 'upper': bounds[1],
                                       'adder': -bounds[0], 'scaler': 1./(bounds[1] - bounds[0])}})
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

                if name in self.coupling_var_cons.values():
                    for key, value in self.coupling_var_cons.items():
                        if name == value:
                            con['equals'] = np.zeros(self.variable_sizes[key])
                            break
                else:
                    # Obtain the reference value of the constraint
                    constr_ref = convar.find('referenceValue')  # type: etree._Element
                    if constr_ref is not None:
                        ref = parse_cmdows_value(constr_ref)
                        if isinstance(ref, str):
                            raise ValueError('referenceValue for constraint "%s" is not numerical' % name)
                        elif not self.does_value_fit(name, ref):
                            warnings.warn('incompatible size of constraint "%s". Will assume the same for all.' % name)
                            ref = np.ones(self.variable_sizes[name]) * np.atleast_1d(ref)[0]
                    else:
                        warnings.warn('no referenceValue given for constraint "%s". Default is all zeros.' % name)
                        ref = np.zeros(self.variable_sizes[name])

                    # Process the constraint type
                    constr_type = convar.find('constraintType')
                    if constr_type is not None:
                        if constr_type.text == 'inequality':
                            constr_oper = convar.find('constraintOperator')
                            if constr_oper is not None:
                                oper = constr_oper.text
                                if oper == '>=' or oper == '>':
                                    con['lower'] = ref
                                elif oper == '<=' or oper == '<':
                                    con['upper'] = ref
                                else:
                                    raise ValueError('invalid constraintOperator "%s" for constraint "%s"' % (oper, name))
                            else:
                                warnings.warn(
                                    'no constraintOperator given for inequality constraint. Default is "&lt;=".')
                                con['upper'] = ref
                        elif constr_type.text == 'equality':
                            if convar.find('constraintOperator') is not None:
                                warnings.warn('constraintOperator given for an equalityConstraint will be ignored')
                            con['equals'] = ref
                        else:
                            raise ValueError('invalid constraintType "%s" for constraint "%s".' % (constr_type.text, name))
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
        if objvars is None:
            raise Exception('cmdows does not contain (valid) objective variables')
        if len(objvars) > 1:
            raise Exception('cmdows contains multiple objectives, but this is not supported')

        return xpath_to_param(objvars[0].find('parameterUID').text)

    @CachedProperty
    def coupled_group(self):
        # type: () -> Optional[Group]
        """:obj:`Group`, optional: Group wrapping the coupled blocks with a converger specified in the CMDOWS file.

        If no coupled blocks are specified in the CMDOWS file this property is `None`.
        """
        if self.coupled_blocks:
            coupled_group = Group()
            for uid in self.coupled_blocks:
                # Get the correct DisciplineComponent
                discipline_component = self.discipline_components[uid]

                # Change input variable names if they are provided as copies of coupling variables
                promotes = ['*']
                if not self.has_converger:
                    for i in discipline_component.inputs_from_xml.keys():
                        if i in self.coupling_vars:
                            promotes.append((i, self.coupling_vars[i]['copy']))

                # Add the DisciplineComponent to the group
                coupled_group.add(uid, self.discipline_components[uid], promotes)

            # Find the convergence type of the coupled group
            if self.has_converger:
                conv_type = self.elem_problem_def.find('problemFormulation/convergerType').text
                if conv_type == 'Gauss-Seidel':
                    coupled_group.ln_solver = LinearGaussSeidel()
                    coupled_group.ln_solver.options['maxiter'] = 10
                    coupled_group.nl_solver = NLGaussSeidel()
                else:
                    raise RuntimeError('Specified convergerType "%s" is not supported.' % conv_type)
            else:
                pass
                # coupled_group.linear_solver = LinearRunOnce()
                # coupled_group.nonlinear_solver = NonLinearRunOnce()
            return coupled_group
        return None

    @CachedProperty
    def consistency_constraint_group(self):
        # type: () -> Optional[Group]
        """:obj:`Group`, optional: Group containing ExecComps for the consistency constraints."""
        elem_ccf = self.elem_arch_elems.find('executableBlocks/consistencyConstraintFunctions')
        if elem_ccf is not None:
            group = Group()
            for child in elem_ccf:
                uid = child.attrib['uID']
                xpaths = []
                for value in self.elem_cmdows.xpath(
                                r'workflow/dataGraph/edges/edge[toExecutableBlockUID="{}"]/toParameterUID/text()'.format(uid)):
                    if 'architectureNodes' not in value and value not in xpaths:
                        xpaths.append(value)

                        name = xpath_to_param(value)
                        size = self.variable_sizes[name]
                        coupling_var = self.coupling_vars[name]

                        group.add(
                            'Gc_ ' + name,
                            ExecComp('{} = {}/{} - 1.'.format(coupling_var['con'], coupling_var['copy'], name),
                                     g=np.zeros(size),
                                     y1=np.zeros(size),
                                     y2=np.zeros(size)),
                            ['*'])
            return group
        return None

    @CachedProperty
    def coordinator(self):
        # type: () -> IndepVarComp
        """:obj:`IndepVarComp`: An `IndepVarComp` representing the system's ``Coordinator`` block.

        This `IndepVarComp` takes care of all system input parameters and initial values of design variables.
        """
        # Add design variables
        _vars = []
        for name, value in self.design_vars.items():
            _vars.append((name, value['initial']))

        # Add system constants
        for name, shape in self.system_inputs.items():
            if name not in self.design_vars.keys():
                if shape == 1:
                    _vars.append((name, 0.))  # TODO: val = ?
                else:
                    _vars.append((name, np.zeros(shape)))  # TODO: val = ?

        return IndepVarComp(_vars)

    @CachedProperty
    def root(self):
        # type: () -> Group
        """:obj:`Group`: The root Group of the Problem."""
        root = Group()
        root.deriv_options['type'] = 'fd'
        root.deriv_options['form'] = 'forward'
        root.deriv_options['step_size'] = 1.0e-4
        root.deriv_options['step_calc'] = 'relative'

        # Add the coordinator
        self.add_subsystem('coordinator', self.coordinator, ['*'])

        # Add all pre-coupling and post-coupling components
        for name, component in self.discipline_components.items():
            if name not in self.coupled_blocks:
                root.add(name, component, ['*'])

        # Add the coupled group
        if self.coupled_group is not None:
            root.add('coupled_group', self.coupled_group, ['*'])

        # Add the consistency constraint group
        if self.consistency_constraint_group is not None:
            root.add('consistency_constraints', self.consistency_constraint_group, ['*'])

        # Put the blocks in the correct order
        root.set_order(list(self.system_order))

        return root

    @root.setter
    def root(self, value):
        pass

    @property
    def driver(self):
        # type: () -> Driver
        """:obj:`Driver`: The Driver of this Problem."""
        return self._driver

    @driver.setter
    def driver(self, driver):
        # type: (Driver) -> None
        if driver is not None:
            # Add the design variables
            for name, value in self.design_vars.items():
                driver.add_desvar(name, lower=value['lower'], upper=value['upper'],
                                  adder=value['adder'], scaler=value['scaler'])

            # Add the constraints
            for name, value in self.constraints.items():
                driver.add_constraint(name, lower=value['lower'], upper=value['upper'], equals=value['equals'])

            # Add the objective
            driver.add_objective(self.objective)

        self._driver = driver

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
            param = xpath_to_param(xpath)
            if param in self.root.unknowns:
                self.root.unknowns[param] = value
