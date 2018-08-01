import os

from OpenLEGO_dev_scripts.test_cases.ssbj.ssbjdatabase.create_cmdows_file import create_cmdows_file
from ssbjkadmos.utils.database import deploy, clean

dir_path = os.path.dirname(os.path.realpath(__file__))

# TODO: Add renaming of input and output files...

clean(dir_path)

deploy(dir_path)

for file in os.listdir(os.path.dirname(__file__)):
    checks = ['Aerodynamics-', 'Performance-', 'Propulsion-', 'Structures-']
    replaces = ['AeroAnalysis-', 'PerformanceAnalysis-', 'PropulsionAnalysis-', 'StructuralAnalysis-']
    for i, check in enumerate(checks):
        if check in file:
            os.rename(file, replaces[i] + file[len(check):])

os.remove('Constraints-input.xml')
os.remove('Constraints-output.xml')
os.remove('Objective-input.xml')
os.remove('Objective-output.xml')

create_cmdows_file()