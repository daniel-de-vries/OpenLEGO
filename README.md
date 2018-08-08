# OpenLEGO

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1f32ed0b023e4d2db498589983652773)](https://www.codacy.com/app/danieldevries6/OpenLEGO?utm_source=github.com&utm_medium=referral&utm_content=daniel-de-vries/OpenLEGO&utm_campaign=badger)

Welcome to the Git repo of *OpenLEGO*!

*OpenLEGO* stands for "Open-source Link between AGILE and OpenMDAO", which is just what it does. It links together two
major efforts in the field of Multidisciplinary Design Analysis and Optimization (MDAO) which are both aming to make
MDAO more readily applicable in the engineering industry.

## Outline

*OpenLEGO* enables to user to combine the strengths of  *KADMOS* and *OpenMDAO*. 
By using *OpenLEGO* the user can manipulate MDAO problems from a high, abstract level using *KADMOS* and
then automatically generate and run an *OpenMDAO* problem from it.

## Installation

The *OpenLEGO* package can be installed using the `setup.py` file provided in the root of this repo. Any required packages
should automatically be installed alongside it.

The following packages are required by *OpenLEGO*:

- [KADMOS](https://pypi.python.org/pypi/kadmos)
- [OpenMDAO](https://testpypi.python.org/pypi/openmdao)
- [lxml](https://pypi.python.org/pypi/lxml)
- [numpy](https://pypi.python.org/pypi/numpy)
- [ssbj-kadmos](https://pypi.python.org/pypi/ssbj-kadmos)

*OpenLEGO* is also on [pip](https://pypi.python.org/pypi/openlego), so it can also be installed simply using 
`pip install openlego`.

## Usage

Check out the test cases in the `openlego.test_suite.test_examples` package included in this repo test get an idea
of how OpenLEGO can be used.

## Credits
*OpenLEGO* was originally developed by [Daniël de Vries](www.daniel-de-vries.com) as part
of his MSc thesis at the [TU Delft](https://www.tudelft.nl/). Feel free to post any feedback on the code, suggestions for new features, and general ideas.

NOTE: OpenLEGO was not developed under the umbrella of the AGILE project and did not receive funding from it.
