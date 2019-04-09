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
import os

import numpy as np
from cached_property import cached_property

from openmdao.api import ExplicitComponent, CaseReader
from openmdao.recorders.sqlite_recorder import SqliteRecorder
from openmdao.vectors.vector import Vector

from openlego.core.problem import LEGOProblem
from openlego.utils.general_utils import str_to_valid_sys_name, warn_about_failed_experiments, \
    unscale_value


class SubDriverComponent(ExplicitComponent):
    """Class exposing an interface to use subdriver (or nested) drivers within an OpenMDAO model.
    This nested subdriver appears as an ExplicitComponent in the top-level hierarchy, but configures
    and executes its own LEGOProblem() and LEGOModel() instances inside.

    Parameters
    ----------
        driver_uid : str
            UID of the main driver under consideration.

        cmdows_path : str
            Path to the CMDOWS file.

        kb_path : str
            Path to the knowledge base.

        data_folder : str
            Path to the data folder in which to store all files and output from the problem.

        base_xml_file : str
            Path to a base XML file to be updated with the problem data.

        super_driver_type : str, optional
            Setting whether the component has a superdriver (used to correctly connect full system)

        create_model_view : bool, optional
            Whether or not to create the model view of the subdriver constructed

        open_model_view : bool, optional
            Whether or not to open the model view automatically if it is created

    Returns
    -------
        SubDriverComponent
    """

    def __init__(self, **kwargs):
        # type: () -> None
        """
        Store some bound methods so we can detect runtime overrides.

        Parameters
        ----------
        **kwargs : dict of keyword arguments
            Keyword arguments that will be mapped into the Component options.
        """
        super(SubDriverComponent, self).__init__(**kwargs)
        self._run_count = 0

    def initialize(self):
        # type: () -> None
        """Initialization of the object with the declaration of settings."""
        self.options.declare('driver_uid')
        self.options.declare('cmdows_path')
        self.options.declare('kb_path')
        self.options.declare('data_folder')
        self.options.declare('base_xml_file')
        self.options.declare('super_driver_type', default=None)
        self.options.declare('create_model_view', default=True)
        self.options.declare('open_model_view', default=False)

    def _add_run_count(self):
        # type: () -> None
        """Update the run counter of the component (used for profiling)."""
        self._run_count += 1

    @cached_property
    def prob(self):
        # type: () -> LEGOProblem
        """Create the problem instance of the subdriver component."""
        return LEGOProblem(cmdows_path=self.options['cmdows_path'],
                                    kb_path=self.options['kb_path'],
                                    data_folder=self.options['data_folder'],  # Output directory
                                    base_xml_file=self.options['base_xml_file'],
                                    driver_uid=self.options['driver_uid'])

    def setup(self):
        # type: () -> None
        """Set up of the explicit component object with a nested LEGOProblem as subdriver."""

        # Load settings for superdriver case
        super_driver_type = self.options['super_driver_type']

        # Set subproblem
        p = self.prob

        # Add inputs/outputs
        for input_name, shape in p.model.model_constants.items():
            self.add_input(input_name, shape=shape)

        for input_name, attrbs in p.model.model_super_inputs.items():
            self.add_input(input_name, shape=attrbs['shape'])

        for output_name, shape in p.model.model_super_outputs.items():
            self.add_output(output_name, shape=shape)

        # Declare partials
        if p.model.model_super_outputs and not super_driver_type:
            input_names = list(p.model.model_super_inputs.keys())
            for output_name in p.model.model_super_outputs.keys():
                self.declare_partials(output_name, input_names,
                                      method='fd', step=1e-6, step_calc='abs')

        # Setup
        p.setup()
        p.final_setup()

        # Store (and view?) model
        if self.options['create_model_view'] or self.options['open_model_view']:
            p.store_model_view(open_in_browser=self.options['open_model_view'])

    def compute(self, inputs, outputs):
        # type: (Vector, Vector) -> None
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
        # sort to have outputs first
        sorted_model_super_inputs = sorted(m.model_super_inputs.keys(), reverse=True)
        for input_name in sorted_model_super_inputs:
            if input_name in m.sm_of_training_params.keys():  # Add these inputs as training data
                sm_uid = m.sm_of_training_params[input_name]
                pred_param = m.find_mapped_parameter(input_name,
                                                     m.sm_prediction_inputs[sm_uid] |
                                                     m.sm_prediction_outputs[sm_uid])
                sm_comp = getattr(m, str_to_valid_sys_name(sm_uid))
                if sm_uid not in failed_experiments.keys():
                    failed_experiments[sm_uid] = (None, None)
                sm_comp.options['train:'+pred_param], failed_experiments[sm_uid]\
                    = p.postprocess_experiments(inputs[input_name], input_name,
                                                failed_experiments[sm_uid])
            else:
                p[input_name] = inputs[input_name]

        # Provide message on failed experiments
        warn_about_failed_experiments(failed_experiments)

        # Set initial values of design variables back to original ones (to avoid using values of
        # last run)
        for des_var, attrbs in m.design_vars.items():
            p[des_var] = attrbs['initial']

        # Run the driver
        print('Running subdriver {}'.format(self.options['driver_uid']))
        if 'Sub-Optimizer' not in p.case_reader_path:
            p.driver.cleanup()
            basename, extension = os.path.splitext(p.case_reader_path)
            case_reader_filename = basename + '_loop' + str(self._run_count) + extension
            p.driver.add_recorder(SqliteRecorder(case_reader_filename))
            p.driver.recording_options['includes'] = ['*']
            p.driver.recording_options['record_model_metadata'] = True
            p.driver._setup_recording()
        p.run_driver()
        self._add_run_count()

        # Pull the value back up to the output array
        doe_out_vecs = {}
        for output_name in m.model_super_outputs:
            # Add these outputs as vectors based on DOE driver
            if output_name in m.doe_parameters.keys():
                doe_out_vecs[output_name] = []
            else:
                if not p.driver.fail:
                    outputs[output_name] = p[output_name]
                else:
                    outputs[output_name] = float('nan')

        # If the driver failed (hence, optimization failed), then send message and clean
        if p.driver.fail:
            print('Driver run failed!')
            p.clean_driver_after_failure()

        # Provide DOE output vectors as output of the component, if this is expected
        if doe_out_vecs:
            # First read out the case reader
            cr = CaseReader(case_reader_filename)
            cases = cr.list_cases('driver')
            for n in range(len(cases)):
                cr_outputs = cr.get_case(n).outputs
                doe_param_matches = {}
                for output_name in doe_out_vecs.keys():
                    doe_param_matches[output_name] = doe_param_match \
                        = m.find_mapped_parameter(output_name, cr_outputs.keys())
                    doe_out_vecs[output_name].append(cr_outputs[doe_param_match][0])

            # Then write the final vectors to the global output array
            for output_name in doe_out_vecs.keys():
                if output_name in p.doe_samples[p.driver_uid]['inputs']:
                    des_var_match = m.find_mapped_parameter(output_name, m._design_vars.keys())
                    doe_out_vecs[output_name] = unscale_value(doe_out_vecs[output_name],
                                                              m._design_vars[des_var_match]['ref0'],
                                                              m._design_vars[des_var_match]['ref'])
                outputs[output_name] = np.array(doe_out_vecs[output_name])
