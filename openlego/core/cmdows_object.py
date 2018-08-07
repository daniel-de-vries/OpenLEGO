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

This file contains the definition of the `CMDOWSObject` class.
"""
from __future__ import absolute_import, division, print_function

from typing import Any, Optional

from openlego.utils.general_utils import CachedProperty


class CMDOWSObject(object):
    """A class that depends on a CMDOWS file and a Knowledge Base.

    This class is used as a mixin for the `LEGOModel` and `LEGOProblem` classes to avoid code dependency.

    Attributes
    ----------
        cmdows_path
        kb_path

        data_folder : str, optional
            Path to the folder in which to store all data generated during the `Problem`'s execution.

        base_xml_file : str, optional
            Path to an XML file which should be kept up-to-date with the latest data describing the problem.
    """

    def __init__(self, cmdows_path=None, kb_path='', data_folder=None, base_xml_file=None, **kwargs):
        # type: (Optional[str], Optional[str], Optional[str], Optional[str]) -> None
        """Initialize a CMDOWS dependent class from a given CMDOWS file and knowledge base.

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
        self._cmdows_path = cmdows_path
        self._kb_path = kb_path
        self.data_folder = data_folder
        self.base_xml_file = base_xml_file

        super(CMDOWSObject, self).__init__()

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
        return super(CMDOWSObject, self).__getattribute__(name)

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

    def invalidate(self):
        # type: () -> None
        """Invalidate the instance.

        All computed (cached) properties will be recomputed upon being read once the instance has been invalidated."""
        for value in self.__class__.__dict__.values():
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