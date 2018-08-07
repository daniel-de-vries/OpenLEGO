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

x_geometry = 'geometry'
x_analyses = 'analyses'

x_z1 = '/'.join([x_root, x_geometry, 'z1'])
x_z2 = '/'.join([x_root, x_geometry, 'z2'])
x_x1 = '/'.join([x_root, x_geometry, 'x1'])

x_y1 = '/'.join([x_root, x_analyses, 'y1'])
x_y2 = '/'.join([x_root, x_analyses, 'y2'])
x_f1 = '/'.join([x_root, x_analyses, 'f'])
x_g1 = '/'.join([x_root, x_analyses, 'g1'])
x_g2 = '/'.join([x_root, x_analyses, 'g2'])

from openlego.test_suite.test_examples.sellar_competences.kb.D1 import D1
from openlego.test_suite.test_examples.sellar_competences.kb.D2 import D2
from openlego.test_suite.test_examples.sellar_competences.kb.F import F
from openlego.test_suite.test_examples.sellar_competences.kb.G1 import G1
from openlego.test_suite.test_examples.sellar_competences.kb.G2 import G2


def list_disciplines():
    return [D1(), D2(), F(), G1(), G2()]


def clean():
    for discipline in list_disciplines():
        os.remove(discipline.in_file)
        os.remove(discipline.out_file)
        os.remove(discipline.partials_file)
    os.remove(base_file_path)


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
