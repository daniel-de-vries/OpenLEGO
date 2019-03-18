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

This file contains code to clean and deploy the knowledge base of the test Sellar case.
"""
from __future__ import absolute_import, division, print_function

import os
import sys
from shutil import copyfile

from openlego.utils.xml_utils import xml_merge

dir_path = os.path.dirname(os.path.realpath(__file__))
base_file_path = os.path.join(dir_path, 'sellar-base.xml')

root_tag = 'dataSchema'
x_root = '/' + root_tag

x_settings = 'settings'
x_variables = 'variables'
x_analyses = 'analyses'

x_a = '/'.join([x_root, x_settings, 'a'])
x_c = '/'.join([x_root, x_settings, 'c'])

x_z1 = '/'.join([x_root, x_variables, 'z1'])
x_z2 = '/'.join([x_root, x_variables, 'z2'])
x_x0 = '/'.join([x_root, x_variables, 'x0'])
x_x1 = '/'.join([x_root, x_variables, 'x1'])

x_y1 = '/'.join([x_root, x_analyses, 'y1'])
x_y2 = '/'.join([x_root, x_analyses, 'y2'])
x_f1 = '/'.join([x_root, x_analyses, 'f'])
x_g1 = '/'.join([x_root, x_analyses, 'g1'])
x_g2 = '/'.join([x_root, x_analyses, 'g2'])

from openlego.test_suite.test_examples.sellar_competences.kb.D_1 import D_1
from openlego.test_suite.test_examples.sellar_competences.kb.D_2 import D_2
from openlego.test_suite.test_examples.sellar_competences.kb.F import F
from openlego.test_suite.test_examples.sellar_competences.kb.G_1 import G_1
from openlego.test_suite.test_examples.sellar_competences.kb.G_2 import G_2
from openlego.test_suite.test_examples.sellar_competences.kb.H import H


def list_disciplines():
    return [D_1(), D_2(), F(), G_1(), G_2(), H()]


def remove_if_exists(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)


def clean():
    for discipline in list_disciplines():
        remove_if_exists(discipline.in_file)
        remove_if_exists(discipline.out_file)
        remove_if_exists(discipline.partials_file)
    remove_if_exists(base_file_path)


def deploy():
    _copy = True
    for discipline in list_disciplines():
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
