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

This file contains code to clean and deploy the knowledge base of the test SSBJ test case.
"""
from __future__ import absolute_import, division, print_function

import os
import sys

from ssbjkadmos.utils.database import deploy as database_deploy
from ssbjkadmos.utils.database import clean as database_clean

dir_path = os.path.dirname(os.path.realpath(__file__))

list_disciplines = ['AeroAnalysis', 'PerformanceAnalysis', 'PropulsionAnalysis', 'StructuralAnalysis']


def deploy():

    database_clean(dir_path)
    database_deploy(dir_path)

    for file in os.listdir(dir_path):
        checks = ['Aerodynamics-', 'Performance-', 'Propulsion-', 'Structures-']
        replaces = ['AeroAnalysis-', 'PerformanceAnalysis-', 'PropulsionAnalysis-', 'StructuralAnalysis-']
        for i, check in enumerate(checks):
            if check in file:
                os.rename(os.path.join(dir_path, file), os.path.join(dir_path, replaces[i] + file[len(check):]))

    os.remove(os.path.join(dir_path, 'Constraints-input.xml'))
    os.remove(os.path.join(dir_path, 'Constraints-output.xml'))
    os.remove(os.path.join(dir_path, 'Objective-input.xml'))
    os.remove(os.path.join(dir_path, 'Objective-output.xml'))


def clean():
    for discipline in list_disciplines:
        for pf in ['-input.xml', '-output.xml', '-partials.xml']:
            os.remove(os.path.join(dir_path, discipline + pf))
    os.remove(os.path.join(dir_path, 'SSBJ-base.xml'))

    for file in os.listdir(dir_path):
        if '__test__' in file:
            os.remove(os.path.join(dir_path, file))
        if '__run__' in file and '_output.xml' in file:
            os.remove(os.path.join(dir_path, file))


if __name__ == '__main__':
    if len(sys.argv) == 1:
        deploy()
    elif len(sys.argv) == 2 and sys.argv[1] == 'clean':
        clean()
