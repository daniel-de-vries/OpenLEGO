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
import warnings

import numpy as np
from openmdao.api import ExplicitComponent, CaseReader

from openlego.utils.general_utils import str_to_valid_sys_name, warn_about_failed_experiments, \
    denormalize_vector


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
        self.options.declare('super_driver_type', default=None)
        self.options.declare('create_model_view', default=True)
        self.options.declare('open_model_view', default=False)

    def setup(self):
        """Setup of the explicit component object with a nested LEGOProblem as subdriver."""

        # Load settings for superdriver case
        super_driver_type = self.options['super_driver_type']

        # Set subproblem
        from openlego.core.problem import LEGOProblem
        p = self.prob = LEGOProblem(cmdows_path=self.options['cmdows_path'],
                                    kb_path=self.options['kb_path'],
                                    data_folder=self.options['data_folder'],  # Output directory
                                    base_xml_file=self.options['base_xml_file'],
                                    driver_uid=self.options['driver_uid'])
        if p.driver_uid == 'Sys-Optimizer':  # TODO: used for testing purposes, remove later!
            p.driver.options['debug_print'] = ['desvars', 'nl_cons', 'ln_cons', 'objs']  # Set printing of debug info

        # Add inputs/outputs
        for input_name, shape in p.model.model_constants.items():
            self.add_input(input_name, shape=shape)

        for input_name, attrbs in p.model.model_super_inputs.items():
            self.add_input(input_name, shape=attrbs['shape'])

        for output_name, shape in p.model.model_super_outputs.items():
            self.add_output(output_name, shape=shape)

        # Declare partials
        if p.model.model_super_outputs and not super_driver_type:
            self.declare_partials('*', '*', method='fd', step=1e-4, step_calc='abs')

        # Setup
        p.initialize()
        p.final_setup()

        # Store (and view?) model
        if self.options['create_model_view'] or self.options['open_model_view']:
            p.store_model_view(open_in_browser=self.options['open_model_view'])

    def compute(self, inputs, outputs):
        """Computation performed by the component.

        Parameters
        ----------
        inputs : all inputs coming from outside the component in the group
        outputs : all outputs provided outside the component in the group"""

        # Define problem of subdriver
        p = self.prob
        m = p.model

        # Push global inputs down
        for input_name in m.model_constants:
            p[input_name] = inputs[input_name]

        failed_experiments = {}
        sorted_model_super_inputs = sorted(m.model_super_inputs.keys(), reverse=True)  # sort to have outputs first
        for input_name in sorted_model_super_inputs:
            if input_name in m.sm_of_training_params.keys():  # Add these inputs as training data for SM
                sm_uid = m.sm_of_training_params[input_name]
                pred_param = m.find_mapped_parameter(input_name,
                                                     m.sm_prediction_inputs[sm_uid] | m.sm_prediction_outputs[sm_uid])
                sm_comp = getattr(m, str_to_valid_sys_name(sm_uid))
                if sm_uid not in failed_experiments.keys():
                    failed_experiments[sm_uid] = (None, None)
                sm_comp.options['train:'+pred_param], failed_experiments[sm_uid]\
                    = p.postprocess_experiments(inputs[input_name], input_name, failed_experiments[sm_uid])
            else:
                p[input_name] = inputs[input_name]

        # Provide message on failed experiments
        warn_about_failed_experiments(failed_experiments)

        # Set initial values of design variables back to original ones (to avoid using values of last run)
        for des_var, attrbs in m.design_vars.items():
            p[des_var] = attrbs['initial']

        # Run the driver
        print('Running subdriver {}'.format(self.options['driver_uid']))
        p.run_driver()

        # Pull the value back up to the output array
        doe_output_vectors = {}
        for output_name in m.model_super_outputs:
            if output_name in m.doe_parameters.keys():  # Add these outputs as vectors based on DOE driver
                doe_output_vectors[output_name] = []
            else:
                if not p.driver.fail:
                    outputs[output_name] = p[output_name]
                else:
                    outputs[output_name] = float('nan')

        # If the driver failed (hence, optimization failed), then send message and clean for next run
        if p.driver.fail:
            print('Driver run failed!')
            p.clean_driver_after_failure()

        if doe_output_vectors:
            # First read out the case reader
            cr = CaseReader(p.case_reader_path)
            cases = cr.list_cases('driver')
            for n in range(len(cases)):
                cr_outputs = cr.get_case(cases[n]).outputs
                doe_param_matches = {}
                for output_name in doe_output_vectors.keys():
                    doe_param_matches[output_name] = doe_param_match = m.find_mapped_parameter(output_name, cr_outputs.keys())
                    doe_output_vectors[output_name].append(cr_outputs[doe_param_match][0])

            # Then write the final vectors to the global output array
            for output_name in doe_output_vectors.keys():
                if output_name in p.doe_samples[p.driver_uid]['inputs']:
                    des_var_match = m.find_mapped_parameter(output_name, m.design_vars.keys())
                    doe_output_vectors[output_name] = denormalize_vector(doe_output_vectors[output_name],
                                                                         m.design_vars[des_var_match]['ref0'],
                                                                         m.design_vars[des_var_match]['ref'])
                outputs[output_name] = np.array(doe_output_vectors[output_name])

        p.cleanup()
