#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2018 I. van Gent and D. de Vries

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains the definition the `SubDriverComponent` class.
"""

from openmdao.api import ExplicitComponent


class SubDriverComponent(ExplicitComponent):
    """Abstract base class exposing an interface to use subdriver (or nested) drivers within an OpenMDAO model. This
    nested subdriver appears as an ExplicitComponent in the top-level hierarchy, but configures and executes its own
    LEGOProblem() and LEGOModel() instances inside.

    Attributes
    ----------
    TODO: Add attributes and complete class description
    """

    def initialize(self):
        self.options.declare('driver_uid')
        self.options.declare('cmdows_path')
        self.options.declare('kb_path')
        self.options.declare('data_folder')
        self.options.declare('base_xml_file')
        self.options.declare('show_model', default=False)

    def setup(self):
        # Add inputs
        # self.add_input()

        # Add outputs
        # self.add_output()

        # Declare partials
        # self.declare_partials('*', '*')

        # Set subproblem
        from openlego.core.problem import LEGOProblem
        p = self.prob = LEGOProblem(cmdows_path=self.options['cmdows_path'],
                                    kb_path=self.options['kb_path'],
                                    data_folder=self.options['data_folder'],  # Output directory
                                    base_xml_file=self.options['base_xml_file'],
                                    driver_uid=self.options['driver_uid'])
        p.invalidate()
        p.model.invalidate()

        # Add missing connections (?)

        # Setup
        p.setup()
        # self.prob.final_setup()

        # View model?
        if self.options['show_model']:
            p.store_model_view(open_in_browser=True)

    def compute(self, inputs, outputs):
        # p = self.prob

        # Push global inputs down
        # p['x'] = inputs['x']
        # etc.

        # Run the driver
        # p.run_driver()

        # Pull the value back up to the output array
        # outputs['y'] = p['y']
        pass
