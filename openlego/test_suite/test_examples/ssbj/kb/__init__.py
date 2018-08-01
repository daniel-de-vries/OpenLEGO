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
        for pf in ['-input.xml', '-output.xml']:
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
