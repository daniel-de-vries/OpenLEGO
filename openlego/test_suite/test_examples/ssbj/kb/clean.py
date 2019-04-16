import os

from ssbjkadmos.utils.database import clean, try_to_remove

# TODO: Add renamed input and output files...

list_disciplines = ['AeroAnalysis', 'PerformanceAnalysis', 'PropulsionAnalysis', 'StructuralAnalysis', 'DpdxAnalysis']

for discipline in list_disciplines:
    try_to_remove('{}{}'.format(discipline, '-input.xml'))
    try_to_remove('{}{}'.format(discipline, '-output.xml'))
    try_to_remove('{}{}'.format(discipline, '-partials.xml'))
base_file_path = 'SSBJ-base.xml'
try_to_remove(base_file_path)

for file in os.listdir(os.path.dirname(__file__)):
    if '__test__' in file:
        os.remove(file)
    if '__run__' in file and '_output.xml' in file:
        os.remove(file)
    if '__cmdows__' in file:
        os.remove(file)