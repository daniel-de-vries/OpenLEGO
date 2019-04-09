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

This file contains the definition the `B2kSolver` class.
"""
import copy
import os
import pickle
from math import isnan as isnan
from typing import List

import numpy as np
import plotly
import plotly.graph_objs as go

from openmdao.api import NonlinearBlockGS
from openmdao.core.analysis_error import AnalysisError
from openmdao.recorders.recording_iteration_stack import Recording
from openmdao.utils.general_utils import format_as_float_or_array

from openlego.utils.general_utils import str_to_valid_sys_name, unscale_value, scale_value
from openlego.utils.xml_utils import param_to_xpath


class B2kSolver(NonlinearBlockGS):
    """Implements a special solver for the BLISS-2000 (B2k) MDO architecture. This solver acts like
       a NonlinearBlockGS solver in terms of executing the system, but assesses convergence
       differently (based on the objective value of the system optimizer in the B2k system) and
       after each system-level optimization, the solver adjusts the bounds of the design variables
       based on the optimum found by the system-level optimization.

       In addition, the class provides specialized outputs (log statements and plotly plots)
       to inspect the optimization during execution.

        Parameters
        ----------
            maxiter : int, optional
                Maximum amount of iterations

            atol : float, optional
                Absolute error tolerance

            rtol : float, optional
                Relative error tolerance

            iprint : int, optional
                Output printing setting for standard OpenMDAO output

            f_k_red : float, optional
                K-factor reduction for design variables

            f_int_inc : float, optional
                Percentage of interval increase if design variable bound is hit

            f_int_inc_abs : float, optional
                Absolute interval increase, minimum increase if percentual increase is too small

            f_int_range : float, optional
                Minimal range of the design variable interval

            print_last_iteration : bool, optional
                Set to True to print the last iteration information in log

            plot_history : bool, optional
                Set to True to plot history of optimizer with plotly

        Returns
        -------
            B2kSolver
        """

    SOLVER = 'NL: BLISS-2000'

    def __init__(self, **kwargs):
        # type: () -> None
        """Initialize the solver and create dictionaries to store results."""
        super(B2kSolver, self).__init__(**kwargs)

        # Dictionaries used to store results
        self.GLOBAL_BOUNDS = {}
        self.LOCAL_BOUNDS = {}
        self.OPT_DV_VALUES = {}
        self.REF0_VALUES = {}
        self.REF_VALUES = {}
        self.OPT_OBJ_VALUES = {}
        self.OPT_CON_VALUES = {}
        self.ATTRBS_CON_VALUES = {}

    @property
    def system_optimizer_name(self):
        # type: () -> str
        """Get the name of the system-level optimizer."""
        sys = self._system
        for sd in sys.super_drivers:
            if sys.loop_element_details[sd] == 'optimizer':
                return sd

    @property
    def system_optimizer_prob(self):
        # type: () -> LEGOProblem
        """Get the system-level optimizer object."""
        return getattr(self._system, str_to_valid_sys_name(self.system_optimizer_name)).prob

    @property
    def system_doe_names(self):
        # type: () -> set
        """Get the system-level DOEs."""
        _system_does = set()
        sys = self._system
        for sd in sys.super_drivers:
            if sys.loop_element_details[sd] == 'doe':
                _system_does.add(sd)
        return _system_does

    @property
    def system_doe_probs(self):
        # type: () -> List[LEGOProblem]
        """Get the system-level DOE LEGOProblem objects"""
        return [getattr(self._system, str_to_valid_sys_name(x)).prob for x in self.system_doe_names]

    def _declare_options(self):
        # type: () -> None
        """Declare options before kwargs are processed in the init method.
        """
        super(B2kSolver, self)._declare_options()

        self.options.declare('print_last_iteration', types=bool, default=True,
                             desc='set to True to print last iteration information in log')
        self.options.declare('plot_history', types=bool, default=True,
                             desc='set to True to plot history of optimizer with plotly')
        self.options.declare('f_k_red', default=2., types=float,
                             desc='K-factor reduction for design variable ranges.')
        self.options.declare('f_int_inc', default=0.25, types=float,
                             desc='Percentage of interval increase if bound it hit.')
        self.options.declare('f_int_inc_abs', default=.1, types=float,
                             desc='Absolute interval increase, minimum increase if percentual '
                                  'increase is too small.')
        self.options.declare('f_int_range', default=1.e-3, types=float,
                             desc='Minimal range of the design variable interval')

    def _iter_initialize(self):
        # type: () -> (float, float)
        """
        Perform any necessary pre-processing operations.

        Returns
        -------
        float
            initial error.
        float
            error at the first iteration.
        """
        if self.options['debug_print']:
            self._err_cache['inputs'] = self._system._inputs._copy_views()
            self._err_cache['outputs'] = self._system._outputs._copy_views()

        return float('nan'), float('nan')

    def _run_apply(self):
        # type: () -> None
        """
        Run the appropriate apply method on the system.
        """
        print('Run_apply for B2k solver')
        super(B2kSolver, self)._run_apply()

    def _solve(self):
        # type: () -> None
        """
        Run the iterative solver.
        """
        maxiter = self.options['maxiter']
        atol = self.options['atol']
        rtol = self.options['rtol']
        iprint = self.options['iprint']

        self._mpi_print_header()

        self._iter_count = 0
        norm0, norm = self._iter_initialize()

        self._norm0 = norm0
        self._objectives = []

        self._mpi_print(self._iter_count, norm, norm0)

        while self._iter_count < maxiter and \
                (norm > atol or isnan(norm)) and (norm0 > rtol or isnan(norm0)):
            with Recording(type(self).__name__, self._iter_count, self) as rec:
                self._single_iteration()
                self._iter_count += 1
                norm0, norm = self._iter_get_norm()
                # With solvers, we want to record the norm AFTER the call, but the call needs to
                # be wrapped in the with for stack purposes, so we locally assign  norm & norm0
                # into the class.
                rec.abs = norm
                rec.rel = norm0

            self._mpi_print(self._iter_count, norm, norm0)
            self._save_history()
            if self.options['print_last_iteration']:
                self._print_last_iteration()
            if self.options['plot_history']:
                self._plot_history()
            self._iter_apply_new_bounds()
            self._untrain_surrogates()

        fail = (np.isinf(norm) or np.isnan(norm) or
                (norm > atol and norm0 > rtol))

        if self._system.comm.rank == 0 or os.environ.get('USE_PROC_FILES'):
            prefix = self._solver_info.prefix + self.SOLVER
            if fail:
                if iprint > -1:
                    msg = ' Failed to Converge in {} iterations'.format(self._iter_count)
                    print(prefix + msg)

                # Raise AnalysisError if requested.
                if self.options['err_on_maxiter']:
                    msg = "Solver '{}' on system '{}' failed to converge."
                    raise AnalysisError(msg.format(self.SOLVER, self._system.pathname))

            elif iprint == 1:
                print(prefix + ' Converged in {} iterations'.format(self._iter_count))
            elif iprint == 2:
                print(prefix + ' Converged')

    def _iter_get_norm(self):
        # type: () -> (float, float)
        """
        Return the norm of the residual.

        Returns
        -------
        float
            norm.
        """
        print('Getting norm for B2k solver')
        prob = self.system_optimizer_prob
        objs = self._objectives
        if not prob.driver.fail:
            objs.append(float(prob[prob.model.objective]))
            if len(objs) > 1:
                norm = abs(objs[-1] - objs[-2])
                norm0 = abs(norm / objs[-2])
            else:
                norm = float('nan')
                norm0 = float('nan')
            return norm0, norm
        else:
            objs.append(float('nan'))
            return float('nan'), float('nan')

    def _iter_apply_new_bounds(self):
        # type: () -> None
        """Apply new bounds on the design variables and set up the models again."""
        prob = self.system_optimizer_prob
        # Get design variables
        for des_var_name, attrbs in prob.model._design_vars.items():
            attrbs_new = self._get_new_bounds(des_var_name, attrbs)
            self._apply_new_bounds(des_var_name, attrbs_new)
        prob.final_setup()
        for doe_prob in self.system_doe_probs:
            doe_prob.final_setup()

    def _get_new_bounds(self, var_name, attrbs):
        # type: (str, dict) -> dict
        """Method that determines new bounds for the design variables for the next BLISS loop.
        Bounds are initially reduced, but will be increased if bounds are hit or if the system-level
        optimization failed.

        Parameters
        ----------
            var_name : str
                Design variable name of which new bounds should be determined
            attrbs : dict
                Attributes of the design variable (ref0, ref, lower, upper, adder, scaler)

        Returns
        -------
            attrbs_new : dict
                Dictionary specifying the new bounds.
        """
        options = self.options
        f_k_red = options['f_k_red']  # K-factor reduction
        f_int_inc = options['f_int_inc']  # fraction of interval increase if bound is hit
        f_int_inc_abs = options['f_int_inc_abs']  # absolute interval increase if fraction too small
        f_int_range = options['f_int_range']  # minimum range of design variable interval

        btol = 1e-2  # Tolerance for boundary hit determination

        prob = self.system_optimizer_prob
        n_loop = self._iter_count - 1
        opt_failed = prob.driver.fail

        # Initialize variables
        val_lb, val_ub = attrbs['lower'], attrbs['upper']
        val_ref0, val_ref = attrbs['ref0'], attrbs['ref']
        adder, scaler = attrbs['adder'], attrbs['scaler']

        def unscaled(x):
            return unscale_value(x, val_ref0, val_ref)

        def scaled(x):
            return scale_value(x, adder, scaler)
        val_opt = scaled(format_as_float_or_array('optimum', copy.deepcopy(prob[var_name])))

        if isinstance(val_opt, np.ndarray) and val_opt.size != 1:
            raise AssertionError('Design vectors are not (yet) supported for the B2k solver.')
        else:
            val_opt = val_opt[0]

        if n_loop == 0:
            local_bounds_pr = None
        else:
            local_bounds_pr = self.LOCAL_BOUNDS[var_name][n_loop - 1]

        # Get and/or store global bounds
        if n_loop == 0:
            model_des_var = prob.model.design_vars[var_name]
            val_min, val_max = scaled(model_des_var['global_lower']), \
                               scaled(model_des_var['global_upper'])
            self.GLOBAL_BOUNDS[var_name] = (val_min, val_max, val_ref0, val_ref)
        else:
            val_min_sc, val_max_sc, val_ref0_gb, val_ref_gb = self.GLOBAL_BOUNDS[var_name]
            val_min_us = unscale_value(val_min_sc, val_ref0_gb, val_ref_gb)
            val_max_us = unscale_value(val_max_sc, val_ref0_gb, val_ref_gb)
            val_min, val_max = scaled(val_min_us), scaled(val_max_us)
        val_interval = val_ub - val_lb

        # If optimum is outside of bounds, then set the bound to the
        out_of_bounds = val_opt < val_lb or val_opt > val_ub
        val_lb_new = val_lb
        val_ub_new = val_ub
        if val_opt < val_lb:
            val_lb_new = val_opt
            val_opt = val_lb
        elif val_opt > val_ub:
            val_ub_new = val_opt
            val_opt = val_ub

        # Reduce bounds based on K-factor reduction
        if not opt_failed and not out_of_bounds:
            adjust = abs((val_ub + val_lb) / 2 - val_opt) / ((val_ub + val_lb) / 2 - val_lb)
            reduce_val = adjust + (1 - adjust) * f_k_red
            val_lb_new = val_opt - (val_interval / reduce_val) / 2
            val_ub_new = val_opt + (val_interval / reduce_val) / 2
        elif n_loop == 0 and opt_failed:
            raise NotImplementedError('First system-level optimization needs to be successful '
                                      'for the BLISS solver to work.')

        # If bound has been hit (twice ==> increase)
        if (n_loop > 0 or opt_failed) and not out_of_bounds:
            val_opt_pr = float(self.OPT_DV_VALUES[var_name][n_loop - 1])
            # lower bound hit twice or optimization failed
            lower_bound_hit = (val_opt - btol <= val_lb and
                               val_opt_pr - btol <= local_bounds_pr[0]) or opt_failed
            dist_lb = abs(val_opt-val_lb)
            # upper bound hit twice or optimization failed
            upper_bound_hit = (val_opt + btol >= val_ub and
                               val_opt_pr + btol >= local_bounds_pr[1]) or opt_failed
            dist_ub = abs(val_opt-val_ub)
            change_bound = []
            if lower_bound_hit and upper_bound_hit:
                if dist_lb < dist_ub:
                    change_bound = ['lb']
                elif dist_ub < dist_lb:
                    change_bound = ['ub']
                else:
                    change_bound = ['lb', 'ub']
            elif lower_bound_hit or upper_bound_hit:
                if upper_bound_hit:
                    change_bound = ['ub']
                else:
                    change_bound = ['lb']
            incr = abs(val_interval * f_int_inc / 2)
            if 'lb' in change_bound:
                if incr >= f_int_inc_abs:
                    val_lb_new = val_lb - val_interval * f_int_inc / 2
                else:
                    val_lb_new = val_lb - f_int_inc_abs
            elif 'ub' in change_bound:
                if incr >= f_int_inc_abs:
                    val_ub_new = val_ub + val_interval * f_int_inc / 2
                else:
                    val_ub_new = val_ub + f_int_inc_abs

        # Check if bounds are not reversed -> otherwise set equal with minimal range
        if val_lb_new > val_ub_new:
            val_lb_new = val_opt - .5*f_int_range
            val_ub_new = val_opt + .5*f_int_range

        # If interval range is smaller than the minimum range -> adjust accordingly
        if abs(val_ub_new - val_lb_new) < f_int_range:
            # First consider upper bound
            dist_ub = abs(val_opt-val_max)
            if dist_ub < .5*f_int_range:
                val_ub_new = val_max
                rest_range_ub = .5*f_int_range-dist_ub
            else:
                val_ub_new = val_opt + .5*f_int_range
                rest_range_ub = 0.
            # Then adjust lower bound accordingly
            dist_lb = abs(val_opt-val_min)
            if dist_lb < .5*f_int_range:
                val_lb_new = val_min
                rest_range_lb = .5*f_int_range-dist_lb
            else:
                val_lb_new = val_opt - .5*f_int_range-rest_range_ub
                rest_range_lb = 0.
            # Add lower bound rest range to the upper bound
            val_ub_new += rest_range_lb

        # If interval is outside maximum bounds -> set equal to appropriate extremum
        if val_lb_new < val_min:
            val_lb_new = val_min
        if val_ub_new > val_max:
            val_ub_new = val_max

        # Save new bounds and nominal values in attribute dictionary
        attrbs_new = {}
        if opt_failed:
            attrbs_new['initial'] = np.array([unscaled((val_ub_new+val_lb_new) / 2.)])
        else:
            attrbs_new['initial'] = np.array([unscaled(val_opt)])
        attrbs_new['lower'], attrbs_new['ref0'] = unscaled(val_lb_new), unscaled(val_lb_new)
        attrbs_new['upper'], attrbs_new['ref'] = unscaled(val_ub_new), unscaled(val_ub_new)

        return attrbs_new

    def _apply_new_bounds(self, var_name, attrbs):
        # type: (str, dict) -> None
        """Apply new bounds on design variables

        Parameters
        ----------
            var_name : str
                Design variable name of which new bounds should be determined
            attrbs : dict
                Attributes of the design variable (ref0, ref, lower, upper, adder, scaler)
        """
        opt_prob = self.system_optimizer_prob
        opt_prob[var_name] = attrbs['initial']
        opt_prob.model.adjust_design_var(var_name,
                                         initial=attrbs['initial'],
                                         lower=attrbs['lower'], upper=attrbs['upper'],
                                         ref=attrbs['ref'], ref0=attrbs['ref0'])
        for doe_prob in self.system_doe_probs:
            for doe_des_var in doe_prob.model.design_vars.keys():
                # TODO: Performance could be drastically improved here by creating an
                # TODO: object with static design variable mappings
                if doe_prob.model.parameter_uids_are_related(param_to_xpath(var_name),
                                                             param_to_xpath(doe_des_var)):
                    doe_prob.model.adjust_design_var(doe_des_var,
                                                     initial=attrbs['initial'],
                                                     lower=attrbs['lower'], upper=attrbs['upper'],
                                                     ref=attrbs['ref'], ref0=attrbs['ref0'])
                    break

    def _untrain_surrogates(self):
        # type: () -> None
        """Untrain the surrogates in the model by setting the train attribute to True."""
        opt_model = self.system_optimizer_prob.model
        for name, component in opt_model.surrogate_model_components.items():
            component.train = True

    def _save_history(self):
        # type: () -> None
        """Save the solvers history to a collection of dictionaries."""

        # Load optimization problem
        prob = self.system_optimizer_prob

        # Save design variables and bounds
        local_bounds = self.LOCAL_BOUNDS
        opt_dv_values = self.OPT_DV_VALUES
        ref0_values = self.REF0_VALUES
        ref_values = self.REF_VALUES
        for var_name, attrbs in prob.model._design_vars.items():

            # Initialize variables
            val_lb, val_ub = attrbs['lower'], attrbs['upper']
            val_ref0, val_ref = attrbs['ref0'], attrbs['ref']
            adder, scaler = attrbs['adder'], attrbs['scaler']

            def scaled(x):
                return scale_value(x, adder, scaler)
            val_opt = scaled(format_as_float_or_array('optimum', copy.deepcopy(prob[var_name])))

            local_bounds.setdefault(var_name, []).append((val_lb, val_ub))
            ref0_values.setdefault(var_name, []).append(val_ref0)
            ref_values.setdefault(var_name, []).append(val_ref)
            opt_dv_values.setdefault(var_name, []).append(val_opt)

        # Save objective value
        var_name = prob.model.objective
        opt_obj_values = self.OPT_OBJ_VALUES
        val_obj = copy.deepcopy(prob[var_name])
        opt_obj_values.setdefault(var_name, []).append(val_obj)

        # Save constraint values
        opt_con_values = self.OPT_CON_VALUES
        attrbs_con_values = self.ATTRBS_CON_VALUES
        for var_name, attrbs in prob.model.constraints.items():
            val_con = copy.deepcopy(prob[var_name])
            opt_con_values.setdefault(var_name, []).append(val_con)
            attrbs_con_values[var_name] = attrbs  # Overwrite allowed: constant for each iteration

        # Pickle all output
        history = dict(local_bounds=local_bounds,
                       opt_dv_values=opt_dv_values,
                       ref0_values=ref0_values,
                       ref_values=ref_values,
                       opt_obj_values=opt_obj_values,
                       opt_con_values=opt_con_values,
                       attrbs_con_values=attrbs_con_values)
        output_folder = prob.model.data_folder
        output_case_str = prob.output_case_string
        output_file_path = os.path.join(output_folder, 'b2k_history_{}.p'.format(output_case_str))
        pickle.dump(history, open(output_file_path, 'wb'))

    def _print_last_iteration(self):
        # type: () -> None
        """Print the results of the last B2k iteration in the log."""
        i = self._iter_count - 1
        print('HISTORY OF LOOP {}'.format(i))
        # Design variables
        print('DESIGN VARIABLES')
        for var_name in sorted(self.OPT_DV_VALUES.keys()):

            def unscaled(x):
                return unscale_value(x, self.REF0_VALUES[var_name][i], self.REF_VALUES[var_name][i])
            print(var_name)
            print('{} < {} < {}  ({})\n'.format(unscaled(self.LOCAL_BOUNDS[var_name][i][0]),
                                                unscaled(self.OPT_DV_VALUES[var_name][i]),
                                                unscaled(self.LOCAL_BOUNDS[var_name][i][1]),
                                                unscaled(self.OPT_DV_VALUES[var_name][i - 1])
                                                if i > 0 else "..."))

        # Constraints
        print('CONSTRAINTS')
        for var_name in sorted(self.OPT_CON_VALUES.keys()):
            print(var_name)
            attrbs = self.ATTRBS_CON_VALUES[var_name]
            if attrbs['equals'] is not None:
                print('{} (== {})\n'.format(self.OPT_CON_VALUES[var_name][i][0], attrbs['equals']))
            elif attrbs['lower'] is not None or attrbs['upper'] is not None:
                print('{} (> {}, < {})\n'.format(self.OPT_CON_VALUES[var_name][i][0],
                                                 attrbs['lower'], attrbs['upper']))
            else:
                print('{} (unbounded)\n')

        # Objectives
        print('OBJECTIVE(S)')
        for var_name in sorted(self.OPT_OBJ_VALUES.keys()):
            print(var_name)
            print('{}\n'.format(self.OPT_OBJ_VALUES[var_name][i][0]))

    def _plot_history(self):
        # type: () -> None
        """Plot the results of the last B2k iteration in the browser."""
        traces_des_vars = []
        des_var_names = sorted(self.OPT_DV_VALUES.keys())
        legend_entries = [x.split('/')[-1] for x in des_var_names]
        for var_name, legend_name in zip(des_var_names, legend_entries):
            ref0s = self.REF0_VALUES[var_name]
            refs = self.REF_VALUES[var_name]
            val_opt = [float(unscale_value(val, ref0s[k], refs[k]))
                       for k, val in enumerate(self.OPT_DV_VALUES[var_name])]
            val_lb = [float(unscale_value(val[0], ref0s[k], refs[k]))
                      for k, val in enumerate(self.LOCAL_BOUNDS[var_name])]
            val_ub = [float(unscale_value(val[1], ref0s[k], refs[k]))
                      for k, val in enumerate(self.LOCAL_BOUNDS[var_name])]
            error_y = []
            error_y_minus = []
            for j in range(len(self.OPT_DV_VALUES[var_name])):
                for l in range(len(val_opt)):
                    if val_ub[l] > val_opt[l]:
                        error_y.append(abs(val_ub[l] - val_opt[l]))
                    else:
                        error_y.append(0.)
                    if val_lb[l] < val_opt[l]:
                        error_y_minus.append(abs(val_lb[l] - val_opt[l]))
                    else:
                        error_y_minus.append(0.)
            trace = go.Scatter(x=list(range(len(val_opt))), y=val_opt, mode='markers',
                               name=legend_name,
                               error_y=dict(type='data', symmetric=False, array=error_y,
                                            arrayminus=error_y_minus),
                               marker=dict(size=12))
            traces_des_vars.append(trace)
        layout_des_vars = go.Layout(title='Design variables of top-level system optimization',
                                    showlegend=True,
                                    xaxis=dict(title='iteration'),
                                    yaxis=dict(title='value'))
        fig_des_vars = go.Figure(data=traces_des_vars, layout=layout_des_vars)
        output_folder = self.system_optimizer_prob.model.data_folder
        output_case_str = self.system_optimizer_prob.output_case_string
        plotly.offline.plot(fig_des_vars,
                            filename=os.path.join(output_folder,
                                                  'b2k_des_vars_{}.html'
                                                  .format(output_case_str)),
                            auto_open=False)

        # Plot constraints
        create_plotly_plot(self.OPT_CON_VALUES,
                           'Constraints of top-level system optimization',
                           'b2k_cons_{}.html'.format(output_case_str),
                           folder=output_folder)

        # Plot objective value(s)
        create_plotly_plot(self.OPT_OBJ_VALUES,
                           'Objective(s) of top-level system optimization',
                           'b2k_obj_{}.html'.format(output_case_str),
                           folder=output_folder)


def create_plotly_plot(dct, plot_title, filename,
                       plot_xaxis_title='iterations',
                       plot_yaxis_title='value',
                       folder='output_files'):
    # type: (dict, str, str, str, str, str) -> None
    """Plot the results of the last B2k iteration in the browser.

    Parameters
    ----------
        dct : dict
            Dictionary with variable names
        plot_title : str
            Title of the plot
        filename : str
            File to be created for the plot
        plot_xaxis_title : str
            Title provided on the X-axis
        plot_yaxis_title : str
            Title provided on the Y-axis
        folder : str
            Destination folder for html file

    Returns
    -------
        Plotly plot stored as html-file.
    """
    traces = []
    var_names = sorted(dct)
    legend_entries = ['/'.join(x.split('/')[-2:]) for x in var_names]
    for var_name, legend_name in zip(var_names, legend_entries):
        values = [float(val) for val in dct[var_name]]
        trace = go.Scatter(x=list(range(len(values))), y=values, mode='markers', name=legend_name,
                           marker=dict(size=12))
        traces.append(trace)
    layout_cons = go.Layout(title=plot_title,
                            showlegend=True,
                            xaxis=dict(title=plot_xaxis_title),
                            yaxis=dict(title=plot_yaxis_title))
    fig_con_vars = go.Figure(data=traces, layout=layout_cons)
    plotly.offline.plot(fig_con_vars,
                        filename=os.path.join(folder, filename),
                        auto_open=False)
