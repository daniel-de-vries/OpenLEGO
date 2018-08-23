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
        driver_uid
        cmdows_path
        kb_path
        data_folder
        base_xml_file
        show_model
    """

    def initialize(self):
        """Initialization of the object with the declaration of settings."""
        self.options.declare('driver_uid')
        self.options.declare('cmdows_path')
        self.options.declare('kb_path')
        self.options.declare('data_folder')
        self.options.declare('base_xml_file')
        self.options.declare('show_model', default=False)

    def setup(self):
        """Setup of the explicit component object with a nested LEGOProblem as subdriver."""

        # Set subproblem
        from openlego.core.problem import LEGOProblem
        p = self.prob = LEGOProblem(cmdows_path=self.options['cmdows_path'],
                                    kb_path=self.options['kb_path'],
                                    data_folder=self.options['data_folder'],  # Output directory
                                    base_xml_file=self.options['base_xml_file'],
                                    driver_uid=self.options['driver_uid'])
        #  p.driver.options['debug_print'] = ['desvars', 'nl_cons', 'ln_cons', 'objs']  # Set printing of debug info

        # Add inputs
        for input_name, shape in p.model.model_constants.items():
            self.add_input(input_name, shape=shape)

        for input_name, attrbs in p.model.model_super_inputs.items():
            self.add_input(input_name, val=attrbs['val'])

        # Add outputs
        for output_name, value in p.model.model_super_outputs.items():
            self.add_output(output_name, val=value)

        # Declare partials
        self.declare_partials('*', '*', method='fd', step=1e-4, step_calc='abs')

        # Setup
        p.setup()
        p.final_setup()

        # Store (and view?) model
        if self.options['show_model']:
            p.store_model_view(open_in_browser=self.options['show_model'])

    def compute(self, inputs, outputs):
        """Computation performed by the component.

        Parameters
        ----------
        inputs : all inputs coming from outside the component in the group
        outputs : all outputs provided outside the component in the group"""

        # Define problem of subdriver
        p = self.prob

        # Push global inputs down
        for input_name in p.model.model_constants:
            p[input_name] = inputs[input_name]

        for input_name in p.model.model_super_inputs:
            p[input_name] = inputs[input_name]

        # Set initial values of design variables back to original ones (to avoid using values of last run)
        for des_var, attrbs in p.model.design_vars.items():
            p[des_var] = attrbs['initial']

        # Run the driver
        print('Optimizing subdriver {}'.format(self.options['driver_uid']))
        p.run_driver()

        # Pull the value back up to the output array
        for output_name in p.model.model_super_outputs:
            outputs[output_name] = p[output_name]

        p.cleanup()
