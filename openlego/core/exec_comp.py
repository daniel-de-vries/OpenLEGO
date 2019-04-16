#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2019 I. van Gent

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains the definition the `ExecComp` class.
"""
from __future__ import absolute_import, division, print_function

from openmdao.api import ExecComp as OpenmdaoExecComp
import time


class ExecComp(OpenmdaoExecComp):
    """Executable component based on mathematical expression with the additional function of adding
    a sleep time to simulate longer execution times."""

    def __init__(self, exprs, sleep_time=None, **kwargs):
        self.number_of_computes = 0
        self.sleep_time = sleep_time
        super(ExecComp, self).__init__(exprs, **kwargs)

    def compute(self, inputs, outputs):
        """
        Execute this component's assignment statements.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.

        outputs : `Vector`
            `Vector` containing outputs.
        """
        OpenmdaoExecComp.compute(self, inputs, outputs)
        self.number_of_computes += 1
        if self.sleep_time is not None:
            time.sleep(self.sleep_time)
