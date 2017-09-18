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

This file contains a reference to the steady lift distribution calculations discipline.
"""
from __future__ import absolute_import, division, print_function

from examples.knowledge_bases.wing_opt.disciplines.dAEDalus import SteadyLiftDistribution


class dSLD(SteadyLiftDistribution):
    pass


if __name__ == '__main__':
    n_ws = 2
    n_lc = 3

    dsld = dSLD(n_ws, n_lc)
    dsld.deploy()
