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

import re
import imp
import sys
import warnings

from lxml import etree
from openmdao.api import Problem, Group, LinearGaussSeidel, NLGaussSeidel, IndepVarComp, Driver
from openmdao.core.problem import _ProbData
from typing import Union, Optional, List

from openlego.AbstractDiscipline import AbstractDiscipline
from openlego.DisciplineComponent import DisciplineComponent
from openlego.xmlutils import xpath_to_param, xml_to_dict
from openlego.util import CachedProperty, parse_string


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
        root
        system_variables
        system_inputs
        driver
        coordinator

        data_folder : str, optional
            Path to the folder in which to store all data generated during the `Problem`'s execution.

        base_xml_file : str, optional
            Path to an XML file which should be kept up-to-date with the latest data describing the problem.
    """

    re_attr_val = re.compile(r'\[((?!\b\d+\b)\b.+\b)\]')

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
        """Ensure both a CMDOWS file and a knowledge base path have been supplied.

        Raises
        ------
            ValueError
                If either no CMDOWS file or no knowledge base path has been supplied
        """
        a = self._cmdows_path is None
        b = self._kb_path is None
        if a or b:
            raise ValueError('No ' + a*'CMDOWS file ' + (a & b)*'and ' + b*'knowledge base path ' + 'specified!')

    def invalidate(self):
        """Invalidate the instance.

        All computed (cached) properties will be recomputed upon being read once the instance has been invalidated."""
        for key, value in self.__class__.__dict__.items():
            if isinstance(value, CachedProperty):
                value.invalidate()

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
    def _cmdows(self):
        # type: () -> etree._Element
        """:obj:`etree._Element`: Root element of the CMDOWS XML file."""
        return etree.parse(self.cmdows_path).getroot()

    @CachedProperty
    def _problem_def(self):
        # type: () -> etree._Element
        """:obj:`etree._Element`: The problemDefition element of this problem's CMDOWS file."""
        return self._cmdows.find('problemDefinition')

    @CachedProperty
    def discipline_components(self):
        # type: () -> dict
        """:obj:`dict`: Dictionary of discipline components by their design competence ``uID`` from CMDOWS.

        Raises
        ------
            RuntimeError
                If a ``designCompetence`` specified in the CMDOWS file does not correspond to an `AbstractDiscipline`.
        """
        _discipline_components = dict()
        for design_competence in self._cmdows.iter('designCompetence'):
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

            _discipline_components.update({uid: DisciplineComponent(cls(),
                                                                    data_folder=self.data_folder,
                                                                    base_file=self.base_xml_file)})
        return _discipline_components

    @CachedProperty
    def block_order(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: List of executable block ``uIDs`` in the order specified in the CMDOWS file."""
        positions = list()
        uids = list()
        for block in self._problem_def.iterfind('problemFormulation/executableBlocksOrder/executableBlock'):
            uid = block.text
            positions.append(int(block.attrib['position']))
            uids.append(uid)
        return [uid for position, uid in sorted(zip(positions, uids))]

    @CachedProperty
    def coupled_blocks(self):
        # type: () -> List[str]
        """:obj:`list` of :obj:`str`: List of ``uIDs`` of the coupled executable blocks specified in the CMDOWS file."""
        _coupled_blocks = []
        for block in self._problem_def.iterfind('problemRoles/executableBlocks/coupledBlocks/coupledBlock'):
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
        return _system_order

    @CachedProperty
    def coupled_group(self):
        # type: () -> Optional[Group]
        """:obj:`Group`, optional: Group wrapping the coupled blocks with a converger specified in the CMDOWS file.

        If no coupled blocks are specified in the CMDOWS file this property is `None`.
        """
        if self.coupled_blocks:
            _coupled_group = Group()
            self.__coupled_group_promotes = []
            for uid in self.coupled_blocks:
                discipline_promotes = self.discipline_components[uid].list_variables()
                _coupled_group.add(uid, self.discipline_components[uid], discipline_promotes)
                self.__coupled_group_promotes.extend(discipline_promotes)

            # Find the convergence type of the coupled group
            conv_type = self._problem_def.find('problemFormulation/convergerType').text
            if conv_type == 'Gauss-Seidel':
                _coupled_group.ln_solver = LinearGaussSeidel()
                _coupled_group.ln_solver.options['maxiter'] = 10
                _coupled_group.nl_solver = NLGaussSeidel()
            else:
                raise RuntimeError('OpenMDAO 1.x only supports a Gauss-Seidel converger.')
            return _coupled_group
        return None

    @CachedProperty
    def root(self):
        """:obj:`Group`: The root `Group` of this `Problem`.

        This `Group` is constructed to represent the system specified in the CMDOWS file. To ensure the root `Group`
        always represents the CMDOWS file it is computed by this class and should not be set directly by the user.

        The `Problem` class, which this class inherits, sets this property to a default, empty `Group` on multiple
        occasions. This behavior is overridden by this class by ignoring any attempts to directly set this property.
        """
        pass

    @root.getter
    def root(self):
        # type: () -> Group
        _root = Group()
        _root.deriv_options['type'] = 'fd'
        _root.deriv_options['form'] = 'forward'
        _root.deriv_options['step_size'] = 1.0e-4
        _root.deriv_options['step_calc'] = 'relative'

        for name, component in self.discipline_components.items():
            if name not in self.coupled_blocks:
                _root.add(name, component, promotes=component.list_variables())

        if self.coupled_group is not None:
            _root.add('coupled_group', self.coupled_group, promotes=self.__coupled_group_promotes)

        return _root

    @root.setter
    def root(self, value):
        # type: (Group) -> None
        pass

    @CachedProperty
    def system_variables(self):
        # type: () -> dict
        """:obj:`dict`: Dictionary of all system variables by their promoted names."""
        self._probdata = _ProbData()
        self.root._init_sys_data('', self._probdata)
        params_dict, unknowns_dict = self.root._setup_variables()
        self._probdata.to_prom_name = self.root._sysdata.to_prom_name
        self._setup_connections(params_dict, unknowns_dict)

        _vars = params_dict.copy()
        _vars.update(unknowns_dict)

        _system_variables = dict()
        for actual, promoted in self._probdata.to_prom_name.items():
            param = _vars[actual]
            del param['pathname']
            if promoted in self._dangling:
                param.update({'system_input': True})
            else:
                param.update({'system_input': False})
            _system_variables.update({promoted: param})

        return _system_variables

    @CachedProperty
    def system_inputs(self):
        # type: () -> dict
        """:obj:`dict`: Dictionary of the system input parameters by their promoted names."""
        return {key: value for key, value in self.system_variables.items() if value['system_input']}

    @CachedProperty
    def _params(self):
        # type: () -> etree._Element
        """:obj:`etree._Element`: The problemRoles/parameters element of the CMDOWS file."""
        params = self._cmdows.find('problemDefinition/problemRoles/parameters')
        if params is None:
            raise Exception('cmdows does not contain (valid) parameters in the problemRoles')
        return params

    @property
    def driver(self):
        # type: () -> Driver
        """:obj:`Driver`: The main `Driver` of this `Problem`.

        The user can replace the `Problem` class' default driver at any point with any other subclass of `Driver`. When
        this is done, the design, objective, and constraint variables will automatically added to it corresponding to
        the CMDOWS file.
        """
        return self._driver

    @driver.setter
    def driver(self, driver):
        # type: (Driver) -> None
        if driver is not None:
            # Process design variables
            desvars = self._params.find('designVariables')
            if desvars is None:
                raise Exception('cmdows does not contain (valid) design variables')

            for desvar in desvars:
                name = xpath_to_param(desvar.find('parameterUID').text)
                bounds = 2 * [None]     # type: List[Optional[str]]

                limit_range = desvar.find('validRanges/limitRange')
                if limit_range is not None:
                    for index, bnd, in enumerate(['minimum', 'maximum']):
                        elem = limit_range.find(bnd)
                        if elem is not None:
                            bounds[index] = parse_string(elem.text)
                driver.add_desvar(name, lower=bounds[0], upper=bounds[1])

            # Process objective variables
            objvars = self._params.find('objectiveVariables')
            if objvars is None:
                raise Exception('cmdows does not contain (valid) objective variables')
            if len(objvars) > 1:
                raise Exception('cmdows contains multiple objectives, but this is not supported')

            obj_name = xpath_to_param(objvars[0].find('parameterUID').text)
            driver.add_objective(obj_name)

            # Process constraint variables
            convars = self._params.find('constraintVariables')
            for convar in convars:
                name = xpath_to_param(convar.find('parameterUID').text)

                # Obtain the reference value of the constraint
                constr_ref = convar.find('referenceValue')
                if constr_ref is not None:
                    ref = parse_string(constr_ref.text)
                    if type(ref) == str:
                        raise ValueError('referenceValue for constraint "%s" is not numerical' % name)
                else:
                    warnings.warn('no referenceValue given for constraint "%s". Default is 0.' % name)
                    ref = 0.

                # Process the constraint type
                constr_type = convar.find('constraintType')
                if constr_type is not None:
                    if constr_type.text == 'inequality':
                        constr_oper = convar.find('constraintOperator')
                        if constr_oper is not None:
                            oper = constr_oper.text
                            if oper == '>=' or oper == '>':
                                driver.add_constraint(name, lower=ref)
                            elif oper == '<=' or oper == '<':
                                driver.add_constraint(name, upper=ref)
                            else:
                                raise ValueError('invalid constraintOperator "%s" for constraint "%s"' % (oper, name))
                        else:
                            warnings.warn('no constraintOperator given for inequality constraint. Default is "&lt;=".')
                            driver.add_constraint(name, upper=ref)
                    elif constr_type.text == 'equality':
                        if convar.find('constraintOperator') is not None:
                            warnings.warn('constraintOperator given for an equalityConstraint will be ignored')
                        driver.add_constraint(name, equals=ref)
                    else:
                        raise ValueError('invalid constraintType "%s" for constraint "%s".' % (constr_type.text, name))
                else:
                    warnings.warn('no constraintType specified for constraint "%s". Default is a <= inequality.')
                    driver.add_constraint(name, upper=ref)

        self._driver = driver

    @CachedProperty
    def coordinator(self):
        # type: () -> Group
        """:obj:`Group`: A `Group` representing the system's ``Coordinator`` block.

        This `Group` takes care of all system input parameters and initial values of design variables.
        """
        _coordinator = Group()
        self.__coordinator_promotes = []
        _names = []
        names = []
        values = []

        # Loop over all edges in the data graph of the CMDOWS file
        for edge in self._cmdows.iterfind('workflow/dataGraph/edges/edge'):
            # Check if this edge departs from the Coordinator block
            from_exec_block_uid = edge.find('fromExecutableBlockUID')
            if from_exec_block_uid is not None and from_exec_block_uid.text == 'Coordinator':
                to_param_uid = edge.find('toParameterUID').text

                # Remove prefix to the XPath relating to copies of variables for initial guesses
                if 'architectureNodes' in to_param_uid:
                    to_param_uid = to_param_uid.split('/')
                    del to_param_uid[2:5]
                    to_param_uid = '/'.join(to_param_uid)

                # Translate CMDOWS' 'fake' uID references to real attribute XPaths and obtain parameter name
                xpath = CMDOWSProblem.re_attr_val.sub(r"[@uID='\1']", to_param_uid)
                param = xpath_to_param(xpath)

                # Check if the parameter is dangling
                if param in self.system_inputs:
                    _names.append('IN_' + re.sub(r'\[.+\]', '', '_'.join(xpath.split('/')[-2:])))
                    values.append(self.system_inputs[param]['val'])

                    n = _names.count(_names[-1])
                    if n > 1:
                        names.append(_names[-1] + '_%d' % n)
                        if n == 2:
                            names[_names.index(_names[-1])] = _names[-1] + '_1'
                    else:
                        names.append(_names[-1])

                    self.__coordinator_promotes.append(param)

        for index, name in enumerate(names):
            # Add an independent variable component for this parameter
            _coordinator.add(name,
                             IndepVarComp(self.__coordinator_promotes[index], val=values[index]),
                             promotes=[self.__coordinator_promotes[index]])

        return _coordinator

    def setup(self, check=True, out_stream=sys.stdout):
        """Setup the `Problem` to prepare it for execution.

        Add the ``Coordinator`` block to the root `Group`, set the order as specified in the CMDOWS file, and setup
        this `Problem` using the super().setup() method.

        See Also
        --------
            openmdao.api.Problem.setup : the super class' `setup()` method.
        """
        # Add the coordinator and set the order of the systems before setting up the problem
        self.root.add('coordinator', self.coordinator, promotes=self.__coordinator_promotes)
        self.root.set_order(list(self.system_order))

        # Setup the problem
        result = super(CMDOWSProblem, self).setup(check, out_stream)

        # Set initial values from the CMDOWS file
        for desvar in self._params.find('designVariables'):
            name = xpath_to_param(desvar.find('parameterUID').text)
            val = desvar.find('nominalValue')
            if val is not None:
                val = parse_string(val.text)
            else:
                warnings.warn('no nominalValue given for designVariable "%s". Default is 0.' % name)
                val = 0.
            self.root.unknowns[name] = val

        return result

    def initialize_from_xml(self, xml):
        # type: (Union[str, etree._ElementTree]) -> None
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
