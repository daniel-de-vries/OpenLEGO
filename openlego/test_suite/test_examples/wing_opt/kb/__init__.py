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

This file contains code to clean and deploy the knowledge base of the test wing optimization case.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
from shutil import copyfile

from openlego.test_suite.test_examples.wing_opt.kb.ConstraintFunctions import ConstraintFunctions
from openlego.test_suite.test_examples.wing_opt.kb.FWE import FWE
from openlego.test_suite.test_examples.wing_opt.kb.ObjectiveFunctions import ObjectiveFunctions
from openlego.test_suite.test_examples.wing_opt.kb.WOM import WOM
from openlego.test_suite.test_examples.wing_opt.kb.dLC import dLC
from openlego.test_suite.test_examples.wing_opt.kb.dSAA import dSAA
from openlego.test_suite.test_examples.wing_opt.kb.dSAMI import dSAMI
from openlego.test_suite.test_examples.wing_opt.kb.dSLD import dSLD
from openlego.test_suite.test_examples.wing_opt.kb.dSSA import dSSA
from openlego.test_suite.test_examples.wing_opt.kb.dSMI import dSMI

from openlego.utils.xml_utils import xml_merge

dir_path = os.path.dirname(os.path.realpath(__file__))
base_file_path = os.path.join(dir_path, 'kb-base.xml')


def list_disciplines(n_ws=2, n_lc=3):
    return [WOM(n_ws),
            dSMI(n_ws, n_lc),
            dSAMI(n_ws, n_lc),
            dSAA(n_ws, n_lc),
            dSSA(n_ws, n_lc),
            dSLD(n_ws, n_lc),
            dLC(n_ws, n_lc),
            FWE(n_ws, n_lc),
            ConstraintFunctions(n_ws),
            ObjectiveFunctions()]


def clean():
    for discipline in list_disciplines():
        os.remove(discipline.in_file)
        os.remove(discipline.out_file)
        os.remove(discipline.json_file)
    os.remove(base_file_path)


def deploy(n_ws=2, n_lc=3):
    _copy = True
    for discipline in list_disciplines(n_ws, n_lc):
        discipline.deploy()
        if _copy:
            _copy = False
            copyfile(discipline.in_file, base_file_path)
        else:
            xml_merge(base_file_path, discipline.in_file)

        xml_merge(base_file_path, discipline.out_file)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        deploy()
    elif len(sys.argv) == 2 and sys.argv[1] == 'clean':
        clean()
