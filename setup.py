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

This file contains the setup code for the OpenLEGO package.
"""
from setuptools import find_packages, setup

from openlego import __version__ as version


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='openlego',
    version=version,
    description='An Open-source link between the AGILE project and the OpenMDAO framework',
    long_description=readme(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='optimization agile multidisciplinary kadmos openmdao engineering xml cpacs',
    url='https://github.com/daniel-de-vries/OpenLEGO',
    download_url='https://github.com/daniel-de-vries/OpenLEGO/tarball/' + version,
    author='Daniël de Vries',
    author_email='danieldevries6@gmail.com',
    license='Apache Software License',
    packages=find_packages(),
    package_data={'openlego.partials': ['partials.xsd']},
    install_requires=[
        'openmdao>=2.4.0',
        'lxml',
        'numpy',
        'ssbjkadmos>=0.1.3',
        'cached-property'
    ],
    include_package_data=True,
    zip_safe=False
)
