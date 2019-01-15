import copy

import numpy as np
import os

from math import isnan as isnan

import plotly
import plotly.graph_objs as go

from openmdao.utils.general_utils import format_as_float_or_array

from openlego.utils.general_utils import str_to_valid_sys_name, unscale_value, scale_value
from openlego.utils.xml_utils import param_to_xpath
from openmdao.core.analysis_error import AnalysisError

from openmdao.api import NonlinearBlockGS
from openmdao.recorders.recording_iteration_stack import Recording


class NonlinearB2kSolver(NonlinearBlockGS):
    SOLVER = 'NL: BLISS-2000'

    # BLISS design variables interval adjustment settings
    # TODO: Add these as options of the solver later
    F_K_RED = 2.0  # K_bound_reduction: K-factor reduction
    F_INT_INC = 0.25  # interval increase: percentage of interval increase if bound is hit
    F_INT_INC_ABS = 0.1  # absolute interval increase: minimum increase if percentual increase is too low
    F_INT_RANGE = 1.e-3  # minimal range of the design variable interval
    PRINT_LAST_ITERATION= True  # Setting to print the results of the last iteration in the log
    PLOT_HISTORY = True  # Setting to plot the history of system-level optimization while running

    # Dictionaries used to store results
    GLOBAL_BOUNDS = {}
    LOCAL_BOUNDS = {}
    OPT_DV_VALUES = {}
    REF0_VALUES = {}
    REF_VALUES = {}
    OPT_OBJ_VALUES = {}
    OPT_CON_VALUES = {}
    ATTRBS_CON_VALUES = {}

    @property
    def system_optimizer_name(self):
        sys = self._system
        for sd in sys.super_drivers:
            if sys.loop_element_details[sd] == 'optimizer':
                return sd

    @property
    def system_optimizer_prob(self):
        return getattr(self._system, str_to_valid_sys_name(self.system_optimizer_name)).prob

    @property
    def system_doe_names(self):
        _system_does = set()
        sys = self._system
        for sd in sys.super_drivers:
            if sys.loop_element_details[sd] == 'doe':
                _system_does.add(sd)
        return _system_does

    @property
    def system_doe_probs(self):
        return [getattr(self._system, str_to_valid_sys_name(x)).prob for x in self.system_doe_names]

    def _iter_initialize(self):
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
        """
        Run the appropriate apply method on the system.
        """
        print('Run_apply for B2k solver')
        super(NonlinearB2kSolver, self)._run_apply()

    def _run_iterator(self):
        # TODO: add docs
        """
        Run the iterative solver.
        """
        # TODO: Get these options from the CMDOWS file.
        maxiter = 30#self.options['maxiter']
        atol = 1e-4#self.options['atol']
        rtol = 1e-4#self.options['rtol']
        iprint = self.options['iprint']

        self._mpi_print_header()

        self._iter_count = 0
        norm0, norm = self._iter_initialize()

        self._norm0 = norm0  # TODO: check where this comes from / is used for?
        self._objectives = []

        self._mpi_print(self._iter_count, norm, norm0)

        while self._iter_count < maxiter and \
                (norm > atol or isnan(norm)) and (norm0 > rtol or isnan(norm0)):
            with Recording(type(self).__name__, self._iter_count, self) as rec:
                self._iter_execute()
                self._iter_count += 1
                norm0, norm = self._iter_get_norm()
                # With solvers, we want to record the norm AFTER the call, but the call needs to
                # be wrapped in the with for stack purposes, so we locally assign  norm & norm0
                # into the class.
                rec.abs = norm
                rec.rel = norm0

            self._mpi_print(self._iter_count, norm, norm0)
            self._save_history()
            if self.PRINT_LAST_ITERATION:
                self._print_last_iteration()
            if self.PLOT_HISTORY:
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

    def solve(self):
        print('Solving with B2k solver')
        super(NonlinearB2kSolver, self).solve()

    def _iter_apply_new_bounds(self):
        # TODO: Add docstring
        prob = self.system_optimizer_prob
        # Get design variables
        print('Apply new bounds for the full iteration')
        for des_var_name, attrbs in prob.model._design_vars.items():
            attrbs_new = self._get_new_bounds(des_var_name, attrbs)  # N.B.: Input attrbs is scaled, output is unscaled
            self._apply_new_bounds(des_var_name, attrbs_new)
        prob.final_setup()  # TODO: or use prob.driver._update_voi_met(prob.model) -> see final_setup() in problem.py
        for doe_prob in self.system_doe_probs:
            doe_prob.final_setup()

    def _get_new_bounds(self, var_name, attrbs):
        # TODO: Update docstring
        """Method that determines new bounds for the design variables for the next BLISS loop. Bounds are initially reduced,
        but will be increased if bounds are hit or if the system-level optimization failed.

        :param des_vars: object containing all design variable details
        :type des_vars: list
        :param z_opt: optimal design vectors
        :type z_opt: dict
        :param f_k_red: K-factor reduction
        :type f_k_red: float
        :param f_int_inc: percentage of interval increase if bound is hit
        :type f_int_inc: float
        :param f_int_inc_abs: absolute interval increase: minimum increase if percentual increase is too low
        :type f_int_inc_abs: float
        :param f_int_range: minimum width of the design variable interval
        :type f_int_range: float
        :param optimization_failed: indication whether optimization was successful
        :type optimization_failed: bool
        :return: enriched design variables object with new bounds
        :rtype: dict
        """
        # TODO: Later take these as options of the solver
        f_k_red = self.F_K_RED
        f_int_inc = self.F_INT_INC
        f_int_inc_abs = self.F_INT_INC_ABS
        f_int_range = self.F_INT_RANGE

        btol = 1e-2  # Tolerance for boundary hit determination

        prob = self.system_optimizer_prob
        n_loop = self._iter_count - 1
        opt_failed = prob.driver.fail

        # Initialize variables
        val_lb, val_ub = attrbs['lower'], attrbs['upper']
        val_ref0, val_ref = attrbs['ref0'], attrbs['ref']
        adder, scaler = attrbs['adder'], attrbs['scaler']
        unscaled = lambda x : unscale_value(x, val_ref0, val_ref)
        scaled = lambda x : scale_value(x, adder, scaler)
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

        # Reduce bounds based on K-factor reduction TODO: add reference
        if not opt_failed and not out_of_bounds:
            adjust = abs((val_ub + val_lb) / 2 - val_opt) / ((val_ub + val_lb) / 2 - val_lb)
            reduce_val = adjust + (1 - adjust) * f_k_red
            val_lb_new = val_opt - ((val_interval) / (reduce_val)) / 2
            val_ub_new = val_opt + ((val_interval) / (reduce_val)) / 2
        elif n_loop == 0 and opt_failed:
            raise NotImplementedError('First system-level optimization needs to be successful '
                                      'for the BLISS solver to work.')

        # If bound has been hit (twice ==> increase)
        if (n_loop > 0 or opt_failed) and not out_of_bounds:
            val_opt_pr = float(self.OPT_DV_VALUES[var_name][n_loop - 1])
            # lower bound hit twice or optimization failed
            lower_bound_hit = (val_opt - btol <= val_lb and val_opt_pr - btol <= local_bounds_pr[0]) \
                              or opt_failed
            dist_lb = abs(val_opt-val_lb)
            # upper bound hit twice or optimization failed
            upper_bound_hit = (val_opt + btol >= val_ub and val_opt_pr + btol >= local_bounds_pr[1]) \
                              or opt_failed
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
        # TODO: Add docstring
        opt_prob = self.system_optimizer_prob
        opt_prob[var_name] = attrbs['initial']
        opt_prob.model.adjust_design_var(var_name,
                                         initial=attrbs['initial'],
                                         lower=attrbs['lower'], upper=attrbs['upper'],
                                         ref=attrbs['ref'], ref0=attrbs['ref0'])
        for doe_prob in self.system_doe_probs:
            for doe_des_var in doe_prob.model.design_vars.keys():
                if doe_prob.model.parameter_uids_are_related(param_to_xpath(var_name),
                                                             param_to_xpath(doe_des_var)):  # TODO: performance could be drastically improved here by creating an object with design variable mappings
                    doe_prob.model.adjust_design_var(doe_des_var,
                                                     initial=attrbs['initial'],
                                                     lower=attrbs['lower'], upper=attrbs['upper'],
                                                     ref=attrbs['ref'], ref0=attrbs['ref0'])
                    break

    def _untrain_surrogates(self):
        # TODO: Add docstring
        opt_model = self.system_optimizer_prob.model
        for name, component in opt_model.surrogate_model_components.items():
            component.train = True

    def _save_history(self):
        # TODO: Update docstring

        # Load optimization problem
        prob = self.system_optimizer_prob

        # Save design variables and bounds
        for var_name, attrbs in prob.model._design_vars.items():

            # Initialize variables
            val_lb, val_ub = attrbs['lower'], attrbs['upper']
            val_ref0, val_ref = attrbs['ref0'], attrbs['ref']
            adder, scaler = attrbs['adder'], attrbs['scaler']
            scaled = lambda x : scale_value(x, adder, scaler)
            val_opt = scaled(format_as_float_or_array('optimum', copy.deepcopy(prob[var_name])))

            LOCAL_BOUNDS = self.LOCAL_BOUNDS
            OPT_DV_VALUES = self.OPT_DV_VALUES
            REF0_VALUES = self.REF0_VALUES
            REF_VALUES = self.REF_VALUES

            if var_name not in LOCAL_BOUNDS:
                LOCAL_BOUNDS[var_name] = [(val_lb, val_ub)]
                REF0_VALUES[var_name] = [val_ref0]
                REF_VALUES[var_name] = [val_ref]
                OPT_DV_VALUES[var_name] = [val_opt]
            else:
                LOCAL_BOUNDS[var_name].append((val_lb, val_ub))
                REF0_VALUES[var_name].append(val_ref0)
                REF_VALUES[var_name].append(val_ref)
                OPT_DV_VALUES[var_name].append(val_opt)

        # Save objective value
        var_name = prob.model.objective
        OPT_OBJ_VALUES = self.OPT_OBJ_VALUES
        val_obj = copy.deepcopy(prob[var_name])
        if var_name not in OPT_OBJ_VALUES:
            OPT_OBJ_VALUES[var_name] = [val_obj]
        else:
            OPT_OBJ_VALUES[var_name].append(val_obj)

        # Save constraint values
        OPT_CON_VALUES = self.OPT_CON_VALUES
        ATTRBS_CON_VALUES = self.ATTRBS_CON_VALUES
        for var_name, attrbs in prob.model.constraints.items():
            val_con = copy.deepcopy(prob[var_name])
            if var_name not in OPT_CON_VALUES:
                OPT_CON_VALUES[var_name] = [val_con]
                ATTRBS_CON_VALUES[var_name] = attrbs
            else:
                OPT_CON_VALUES[var_name].append(val_con)

    def _print_last_iteration(self):
        i = self._iter_count - 1
        print('HISTORY OF LOOP {}'.format(i))
        # Design variables
        print('DESIGN VARIABLES')
        for var_name in sorted(self.OPT_DV_VALUES.keys()):
            unscaled = lambda x : unscale_value(x, self.REF0_VALUES[var_name][i],
                                               self.REF_VALUES[var_name][i])
            print(var_name)
            print('{} < {} < {}  ({})\n'.format(unscaled(self.LOCAL_BOUNDS[var_name][i][0]),
                                                unscaled(self.OPT_DV_VALUES[var_name][i]),
                                                unscaled(self.LOCAL_BOUNDS[var_name][i][1]),
                                                unscaled(self.OPT_DV_VALUES[var_name][i - 1]) if i > 0 else "..."))

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
        # Plot design variables
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
            for j in range(len(self.OPT_DV_VALUES[var_name])):
                error_y = []
                error_y_minus = []
                for l in range(len(val_opt)):
                    if val_ub[l] > val_opt[l]:
                        error_y.append(abs(val_ub[l] - val_opt[l]))
                    else:
                        error_y.append(0.)
                    if val_lb[l] < val_opt[l]:
                        error_y_minus.append(abs(val_lb[l] - val_opt[l]))
                    else:
                        error_y_minus.append(0.)
            trace = go.Scatter(x=range(len(val_opt)), y=val_opt, mode='markers', name=legend_name,
                               error_y=dict(type='data', symmetric=False, array=error_y,
                                            arrayminus=error_y_minus),
                               marker=dict(size=12))
            traces_des_vars.append(trace)
        layout_des_vars = go.Layout(title='Design variables of top-level system optimization',
                                 showlegend=True,
                                 xaxis=dict(title='iteration'),
                                 yaxis=dict(title='value'))
        fig_des_vars = go.Figure(data=traces_des_vars, layout=layout_des_vars)
        plotly.offline.plot(fig_des_vars,
                            filename=os.path.join('output_files', 'ssbj_b2k_des_vars.html'))

        # Plot constraints
        create_plotly_plot(self.OPT_CON_VALUES,
                           'Constraints of top-level system optimization',
                           'ssbj_b2k_cons.html')

        # Plot objective value(s)
        create_plotly_plot(self.OPT_OBJ_VALUES,
                           'Objective(s) of top-level system optimization',
                           'ssbj_b2k_obj.html')


def create_plotly_plot(dct, plot_title, filename,
                       plot_xaxis_title='iterations',
                       plot_yaxis_title='value',
                       folder='output_files'):
    # TODO: Add docstring
    traces = []
    var_names = sorted(dct)
    legend_entries = ['/'.join(x.split('/')[-2:]) for x in var_names]
    for var_name, legend_name in zip(var_names, legend_entries):
        values = [float(val) for val in dct[var_name]]
        trace = go.Scatter(x=range(len(values)), y=values, mode='markers', name=legend_name,
                           marker=dict(size=12))
        traces.append(trace)
    layout_cons = go.Layout(title=plot_title,
                            showlegend=True,
                            xaxis=dict(title=plot_xaxis_title),
                            yaxis=dict(title=plot_yaxis_title))
    fig_con_vars = go.Figure(data=traces, layout=layout_cons)
    plotly.offline.plot(fig_con_vars,
                        filename=os.path.join(folder, filename))
