#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2017 D. de Vries

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
from setuptools import setup, find_packages

version = '0.0.0dev1'

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='openlego',
    version=version,
    description='An Open-source link between the AGILE project and the OpenMDAO framework',
    long_description=readme(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha'
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
    ],
    keywords='optimization agile multidisciplinary kadmos openmdao engineering xml cpacs',
    # TODO: url='https://github.com/some/git/repo',
    # TODO: download_url='https://github.com/some/git/repo/dist/' + version + '.tar.gzip',
    author='Daniël de Vries',
    author_email='danieldevries6@gmail.com',
    license='Apache Software License',
    packages=['openlego'],
    install_requires=[
        'kadmos',
        'openmdao',
        'lxml',
        'numpy',
        'matlab',
        'matplotlib'
    ],
    python_requires='>=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*, <4',
    include_package_data=True,
    zip_safe=False
)
