# OpenLEGO
Welcome to the Git repo of *OpenLEGO*!

*OpenLEGO* stands for "Open-source Link between AGILE and OpenMDAO", which is just what it does. It links together two
major efforts in the field of Multidisciplinary Design Analysis and Optimization (MDAO) which are both aming to make
MDAO more readily applicable in the engineering industry.

## Outline

The [AGILE project](https://www.agile-project.eu/) is an international research collaboration , funded by the European
Commission, with the ultimate goal of reaching:
>"... a speed up of 40% when solving realistic MDO problems compared to today's state-of-the-art."

One way this speed up is envisaged is creating tools which allow engineers to manipulate the analysis and design tools,
connected to one another within a multidisciplinary analysis or optimization architecture, as if they were building
blocks that one could take apart and fit together another way effortlessly - much like LEGO blocks. If they could be
treated as such, it becomes easier for engineers to set-up and reconfigure large, interconnected frameworks of tools.

Within *AGILE* the [KADMOS package](https://pypi.python.org/pypi/kadmos) was developed for this purpose. This
Open-source software, written in Python, uses graph theory to provide tools for automated (re)configuration and
visualization of complex MDAO problems on an abstract level.

As of yet unrelated to the *AGILE* project is the [OpenMDAO framework](http://openmdao.org/). This Open-source platform
provides classes and interfaces wrapping complex data logic and high performance optimization algorithms. It allows
for problems to be decomposed into isolated components and groups of components, which can easily be tied together to
form complex, multidisciplinary analysis and optimization systems.

*OpenLEGO* enables to user to combine the strength of both of these projects by providing a direct link between *KADMOS*
and *OpenMDAO*. By using *OpenLEGO* the user can manipulate MDAO problems from a high, abstract level using *KADMOS* and
then automatically generate and run an *OpenMDAO* problem from it.

## Installation

The *OpenLEGO* package can be installed using the `setup.py` file provided in the root of this repo. Any required packages
should automatically be installed alongside it.

The following packages are required by *OpenLEGO*:

- [KADMOS](https://pypi.python.org/pypi/kadmos)
- [OpenMDAO](https://testpypi.python.org/pypi/openmdao)
- [lxml](https://pypi.python.org/pypi/lxml)
- [numpy](https://pypi.python.org/pypi/numpy)
- [matplotlib](https://pypi.python.org/pypi/matplotlib/)

## Usage

Two example cases have been added in the `example` folder to serve as a tutorials, as well as show cases of *OpenLEGO*:
- **The Sellar problem**: This case can be run by executing `examples/sellar.py`. It sets up and solves the Sellar 
optimization problem. This test case is also described by *KADMOS* and *OpenMDAO*. More information on this problem can
be found at:
    - https://arc.aiaa.org/doi/abs/10.2514/6.1996-714
    - https://www.researchgate.net/publication/2759746_Response_Surface_Based_Concurrent_Subspace_Optimization_For_Multidisciplinary_System_Design
    - http://openmdao.readthedocs.io/en/latest/usr-guide/tutorials/sellar.html
- **A wing optimization problem**: This case can be run by executing `examples/wing_opt.py`. It sets up and solves a very
basic aero-structural wing optimization problem using the Open-source [dAEDalus tool](https://github.com/sbind/dAEDalusNXT).
Note that *dAEDalus* is not included in this repo and should be obtained and installed separately before this example case
can be run.
