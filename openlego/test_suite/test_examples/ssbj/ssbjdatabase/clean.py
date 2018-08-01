import os

from ssbjkadmos.utils.database import clean

# TODO: Add renamed input and output files...

dir_path = os.path.dirname(os.path.realpath(__file__))

clean(dir_path)