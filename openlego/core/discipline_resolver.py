#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2019 J.H. Bussemaker

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains the definition of the `DisciplineResolver` and `ModuleDisciplineResolver` classes.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
import abc
from typing import Union, Optional, Type, List

from openlego.core.abstract_discipline import AbstractDiscipline


class DisciplineResolver(object):
    """
    Abstract class defining an interface for resolving AbstractDiscipline classes for designCompetences.
    """

    @abc.abstractmethod
    def resolve_discipline(self, name, mode):
        # type: (str, str) -> Optional[Union[AbstractDiscipline, Type[AbstractDiscipline]]]
        """
        Return an AbstractDiscipline class or instance if matched to a name and mode.

        :param name: Design competence name (designCompetence/ID node in CMDOWS)
        :param mode: Design competence mode ("main" is the default mode)
        :return:
        """


class DisciplineInstanceResolver(DisciplineResolver):
    """
    Resolves disciplines from a given list of AbstractDiscipline instances.
    Matches by name, or <name>_<mode> if mode != "main".
    """

    def __init__(self, disciplines=None):
        # type: (Optional[List[AbstractDiscipline]]) -> None
        self._disciplines = disciplines or []  # type: List[AbstractDiscipline]

    @property
    def disciplines(self):
        return self._disciplines

    def resolve_discipline(self, name, mode):
        # Adapt name to reflect the mode
        if mode and mode != 'main':
            name = '{}_{}'.format(name, mode)

        # Search for discipline
        for discipline in self.disciplines:
            if discipline.name == name:
                return discipline

    def write_io(self, folder):
        """
        Write input/output (and partials if available) specification files of all disciplines to the provided folder.

        :param folder:
        :return:
        """
        for discipline in self.disciplines:
            discipline.deploy(folder)


class ModuleDisciplineResolver(DisciplineResolver):
    """
    Tries to resolve disciplines by searching for Python modules in a path that match the given name and mode.
    The name of the Python module should be exactly <name>.py or <name>_<mode>.py if mode != "main". Inside the module,
    the AbstractDiscipline subclass should be defined with the same name as the module.
    """

    def __init__(self, path):
        # type: (str) -> None
        self._path = path
        self._is_path = path and os.path.isdir(path)

    def resolve_discipline(self, name, mode):
        if not self._is_path:
            return

        # Adapt name to reflect the mode
        if mode and mode != 'main':
            name = '{}_{}'.format(name, mode)

        try:
            # Try importing for Python 3
            try:
                import importlib.util as importlibutil
                pyfp = os.path.join(os.path.abspath(self._path), name + '.py')
                spec = importlibutil.spec_from_file_location(name, pyfp)
                mod = importlibutil.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules[name] = mod

            # Try importing for Python 2
            except ImportError:
                import imp
                fp, pathname, description = imp.find_module(name, [self._path])
                mod = imp.load_module(name, fp, pathname, description)

            # Get the class from the imported module
            cls = getattr(mod, name)

            # Check the class type
            if not issubclass(cls, AbstractDiscipline):
                return

            # Success!
            return cls

        except Exception:
            pass

        finally:
            if 'fp' in locals():
                fp.close()
