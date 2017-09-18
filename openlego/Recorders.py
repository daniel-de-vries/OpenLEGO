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

This file contains the definitions of ``OpenMDAO`` `Recorder` utility classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from collections import OrderedDict

if sys.version_info[0] == 3:
    import tkinter as tk
else:
    import Tkinter as Tk
    tk = Tk

import abc
import numpy as np
import time
import matplotlib

from abc import abstractmethod
from multiprocessing import Process, Pipe
from openmdao.core.driver import Driver
from openmdao.solvers.solver import Solver
from openmdao.core.system import System
from openmdao.recorders.base_recorder import BaseRecorder
from typing import Optional, Any, Union

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib import pyplot as plt, ticker as ticker
import matplotlib.colors as colors
from matplotlib.figure import Figure
from mpl_toolkits import axisartist as aa
from mpl_toolkits.axes_grid1.parasite_axes import host_subplot_class_factory

from openlego.util import try_hard


class BaseIterationPlotter(BaseRecorder):
    """Base class enabling continually updated plots.

    This is an abstract base class which enables data from an OpenMDAO run to be plotted and for that plot to be updated
    for each iteration/function evaluation. This class uses matplotlib for all the plotting functions. A separate
    process is used to host and manage the plot. This avoids the blocking and stalling behavior of the main loop by
    matplotlib.

    Attributes
    ----------
        save_settings : dict
            Default setting sused when saving the plot as an image file.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        # type: () -> None
        """Initialize the class."""
        super(BaseIterationPlotter, self).__init__()
        self.options.declare('save_on_close', False, desc='Set to True to save figure when closing the Recorder')
        self.save_settings = {'path': self.__class__.__name__ + '_figure.png',
                              'dpi': 600, 'ar': 1.61803398875, 'width': 4000}

        self._in, self._out = Pipe()
        self._callback_in, self._callback_out = Pipe()
        self._process = None

        self._tk = None
        self._canvas = None
        self._toolbar = None
        self._fig = None

        self.options['record_objectives'] = True
        self.options['record_constraints'] = True

    @abstractmethod
    def init_fig(self, fig):
        # type: (Figure) -> None
        """Initialize the figure.

        A plotting recorder implementing this method should use it to setup parts of the figure that should only be
        initialized once, such as titles, labels, and legends.

        Parameters
        ----------
            fig : :obj:`Figure`
                Instance of `Figure` on which the plot should be initialized.
        """
        raise NotImplementedError

    @abstractmethod
    def _update_plot(self, *args):
        # type: (Any) -> None
        """Update the plot with data from the next iteration/function evaluation.

        This method should be implemented to insert new data into the plot. This method is called within the plot
        handling `Process` and should never be called directly.

        Parameters
        ----------
            *args : any
                Data to update or enrich the plot with.
         """
        raise NotImplementedError

    def startup(self, object_requesting_recording):
        # type: (Union[Driver, System, Solver]) -> None
        """Call the `BaseRecorder` `startup()` method and start the `Process` handling the figure.

        Parameters
        ----------
            object_requesting_recording : Union[Driver, System, Solver]
                The Object to which this recorder is attached.
        """
        super(BaseIterationPlotter, self).startup(object_requesting_recording)

        self._process = Process(target=self._process_run)
        self._process.daemon = True
        self._process.start()

    def _process_run(self):
        # type: () -> None
        """Perform the figure handling operations.

        This funciton is the entry point for the plot handling `Process`. It should not be used directly.

        Notes
        -----
            This function is executed within a separate `Process`. Any parameters assigned from within this scope can
            only be accessed by function also running on this `Process`.
        """
        self._tk = tk.Tk()
        self._tk.protocol('WM_DELETE_WINDOW', self._tk.quit())

        def plt_figure():
            # type: () -> Figure
            """Open a `Figure` and wait for a short time.

            This is part of a workaround to ensure a figure window always opens.

            Returns
            -------
                :obj:`Figure`
                    Instance of the openened `Figure`.
            """
            fig = plt.figure()
            time.sleep(1e-4)
            return fig
        self._fig = try_hard(plt_figure, try_hard_limit=4)  # type: Figure

        self._canvas = FigureCanvasTkAgg(self._fig, master=self._tk)
        self._toolbar = NavigationToolbar2TkAgg(self._canvas, self._tk)
        self._toolbar.update()
        self._canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.init_fig(self._fig)

        self._tk.after(1, self._loop())
        self._tk.mainloop()

    def _loop(self):
        # type: () -> None
        """Continually handle the figure on the dedicated `Process`.

        This function is the loop of the `Process` handling the plot. It is executed on the dedicated `Process` handling
        the figure and therefore has access to the parameters that were assigned in `_process_run()`.

        Notes
        -----
            This method should never be called directly. Any instructions to manipulate the figure are communicated
            through `Pipe`s. Convenience methods have been set up for this purpose.
        """
        while self._out.poll():
            out = self._out.recv()
            if out is None or not isinstance(out, tuple):
                raise ValueError('packet sent to _update() is not a tuple')
            elif 'update' in out[0] or 'save' in out[0]:
                if 'update' in out[0]:
                    self._update_plot(*out[1:])
                elif 'save' in out[0]:
                    path, dpi, ar, width = out[1:]
                    self._fig.set_size_inches(ar * width / dpi, ar * width / dpi)
                    self._fig.savefig(path, dpi=dpi)

                if 'block' in out[0]:
                    self._callback_in.send(True)
            elif 'close' in out[0]:
                self._tk.destroy()
                if 'block' in out[0]:
                    self._callback_in.send(True)
                return

        self._canvas.draw()
        self._tk.after(1, self._loop)

    def _blocking_call(self, instr, *args):
        # type: (str, *Any) -> Any
        """Send an instruction to the `Process` handling the plot and wait for a reply.

        Parameters
        ----------
            instr : str
                Instruction to send to the `Process`.

            *args
                Any arguments to send along with the instruction.

        Returns
        -------
            any
                A reply from the `Process`.
        """
        self._in.send(('block %s' % instr,) + args)
        while not self._callback_out.poll():
            time.sleep(1.e-4)
        return self._callback_out.recv()

    def record_metadata_driver(self, object_requesting_recording):
        pass

    def record_metadata_system(self, object_requesting_recording):
        pass

    def record_metadata_solver(self, object_requesting_recording):
        pass

    def record_iteration_driver(self, object_requesting_recording, metadata):
        super(BaseIterationPlotter, self).record_iteration_driver(object_requesting_recording, metadata)

        self._in.send(('update',
                       self._desvars_values,
                       self._responses_values,
                       self._objectives_values,
                       self._constraints_values,
                       metadata))

    def record_iteration_system(self, object_requesting_recording, metadata):
        super(BaseIterationPlotter, self).record_iteration_system(object_requesting_recording, metadata)

        self._in.send(('update',
                       self._inputs,
                       self._outputs,
                       self._resids,
                       metadata))

    def record_iteration_solver(self, object_requesting_recording, metadata, **kwargs):
        super(BaseIterationPlotter, self).record_iteration_solver(object_requesting_recording, metadata, **kwargs)

        self._in.send(('update',
                       self._abs_error,
                       self._rel_error,
                       self._outputs,
                       self._resids,
                       metadata))

    def save_figure(self, path=None, dpi=None, ar=None, width=None):
        # type: (Optional[str], Optional[int], Optional[float], Optional[int]) -> None
        """Save the current plot to an image file.

        Parameters
        ----------
            path : str, optional
                Path of the image file.

            dpi : int, optional
                Resolution of the image file in dots per inch.

            ar : float, optional
                Aspect ratio of the image file.

            width : int, optional
                Width of the image file in pixels.

        Notes
        -----
            If any of the parameters are not given the defaults stored in the `save_settings` dictionary will be used.

            This function makes a blocking call to the `Process` to ensure the figure really is saved once this function
            returns.
        """
        if path is None:
            path = self.save_settings['path']
        if dpi is None:
            dpi = self.save_settings['dpi']
        if ar is None:
            ar = self.save_settings['ar']
        if width is None:
            width = self.save_settings['width']
        self._blocking_call('save', path, dpi, ar, width)

    def save_and_close(self, path=None, dpi=None, ar=None, width=None):
        # type: (Optional[str], Optional[int], Optional[float], Optional[int]) -> None
        """Save the current plot to an image file and close this `Recorder`.

        Parameters
        ----------
            path : str, optional
                Path of the image file.

            dpi : int, optional
                Resolution of the image file in dots per inch.

            ar : float, optional
                Aspect ratio of the image file.

            width : int, optional
                Width of the image file in pixels.

        Notes
        -----
            If any of the parameters are not given the defaults stored in the `save_settings` dictionary will be used.

            This function makes a blocking call to the `Process` to ensure the figure really is saved once this function
            returns.
        """
        self.save_figure(path, dpi, ar, width)

        # Prevent save_on_close option from saving the figure again, then call self.close()
        self.options['save_on_close'] = False
        self.close()

    def close(self):
        """Close the figure, then calls the `BaseRecorder` `close()` method.

        If the option ``save_on_close`` is set to `True`, this function first saves the figure and waits for
        confirmation before it closes it and this `Recorder`.
        """
        # Potentially save the figure before closing
        if self.options['save_on_close']:
            self.save_figure()

        # Close the figure and call super
        self._blocking_call('close')
        super(BaseIterationPlotter, self).close()


class VOIPlotter(BaseIterationPlotter):
    """Specialized `BaseIterationPlotter` plotting several variables of interest with a regular line plot.

    Only those variables specified by settings this `Recorder`'s `includes` property will be plotted. If no variables
    are specified this way an error is thrown.

    Attributes
    ----------
        xdata : :obj:`np.ndarray`
            Numpy array holding the x-data of the plot.

        vois : dict
            Python dictionary representing the information of all variables of interest.

        lines : :obj:`list` of `Line`s
            The `Line` objects of all variable of interest plots.

    """

    def __init__(self):
        # type: () -> None
        """Initializes the `VOIPlotter`."""
        super(VOIPlotter, self).__init__()

        self.options.add_option('legend', [],
                                desc="List of legend entries for each VOI to be plotted. "
                                     "If given, this list needs to have the same length as the options['includes']")
        self.options.add_option('labels', [],
                                desc="List of labels to give to each VOI to be plotted. "
                                     "If given, this list needs to have the same length as the options['includes']")
        self.options['includes'] = []

        self.xdata = None
        self.vois = None

        self.lines = None

    def startup(self, object_requesting_recording):
        # type: (System) -> None
        """Check `includes`, `legend`, and `labels` options for validity and compatibility and create `vois` dictionary.

        Parameters
        ----------
            object_requesting_recording : :obj:`System`
                `Sytem` that owns this `Recorder`.

        Raises
        ------
            AttributeError
                If the `includes`, `legend`, and/or `labels` are not valid or incompatible with one another.
        """
        if not isinstance(object_requesting_recording, System):
            raise ValueError('This Recorder must be attached to a System.')

        includes = self.options['includes']
        legend = self.options['legend']
        labels = self.options['labels']
        if len(includes) == 0 or includes[0] == '*':
            raise AttributeError("At least one variable needs to be put into options['includes']")
        elif len(legend) > 0 and len(legend) != len(includes):
            raise AttributeError('Number of legend entries does not match number of included variables')
        elif len(labels) > 0 and len(labels) != len(includes):
            raise AttributeError('Number of labels does not match number of included variables')

        if len(labels) == 0:
            self.options['labels'] = includes[:]
        if len(legend) == 0:
            self.options['legend'] = legend[:]

        self.vois = OrderedDict()

        super(VOIPlotter, self).startup(object_requesting_recording)

    def init_fig(self, fig):
        # type: (Figure) -> None
        """Initialize the variables of interest figure.

        Parameters
        ----------
            fig : :obj:`Figure`
                Instance of the `Figure` which should be populated.
        """
        host_subplot_class = host_subplot_class_factory(aa.Axes)

        ax_main = host_subplot_class(fig, 111)
        fig.add_subplot(ax_main)
        ax_main.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        offset = 60
        flag = False
        pos = ['left', 'right']

        for index, key in enumerate(self.options['includes']):
            if not index:
                ax = ax_main
                ax.set_xlim([0, 1])
                ax.set_xlabel('Evaluation #')
            else:
                ax = ax_main.twinx()

            ax.autoscale(True, 'y')
            ax.set_ylabel(self.options['labels'][index])

            if index > 1:
                new_fixed_axis = ax.get_grid_helper().new_fixed_axis
                ax.axis[pos[flag]] = new_fixed_axis(loc=pos[flag], axes=ax, offset=((index//2)*offset, 0))
                ax.axis[pos[flag]].toggle(all=True)

            line, = ax.plot([], [], label=self.options['legend'][index])
            color = line.get_color()
            ax.axis[pos[flag]].label.set_color(color)
            ax.spines[pos[flag]].set_color(color)
            ax.tick_params(axis='y', color=color)

            self.vois.update({key: {'ax': ax, 'line': line, 'data': np.array([])}})

            flag ^= True

        ax_main.legend(bbox_to_anchor=(.5, 1.), loc='lower center', ncol=4)

    def _update_plot(self, *args):
        # type: (dict, dict, dict, dict) -> None
        """Insert the new data into the plot and refresh it.

        Parameters
        ----------
            inputs, outputs, resids, metadata : dict
                Dictionaries containing inputs, outputs, residuals, and metadata of the system.
        """
        inputs, outputs, resids, metadata = args

        if self.xdata is None:
            self.xdata = np.array([0.])
        else:
            self.xdata = np.append(self.xdata, [self.xdata[-1] + 1])

        for key in self.vois.keys():
            if key in inputs:
                data = inputs[key]
            elif key in outputs:
                data = outputs[key]
            elif key in resids:
                data = resids[key]
            else:
                raise ValueError('Variable of interest "%s" does not belong to the System holding this Recorder.' % key)

            self.vois[key]['data'] = np.append(self.vois[key]['data'], [data])
            self.vois[key]['line'].set_data(self.xdata, self.vois[key]['data'])
            self.vois[key]['ax'].relim()
            self.vois[key]['ax'].autoscale_view()


class SimpleObjectivePlotter(BaseIterationPlotter):
    """Specialized `BaseIterationPlotter` which simply plots the normalized objective function value against iteration.

    Attributes
    ----------
        obj_name : str
            Name of the objective function value.

        obj_init : float
            Initial value of the objective function

        xdata, ydata : :obj:`np.ndarray`
            The x- and y-data of the plot.

        ax : :obj:`Axes`
            The axis of the plot.

        line : :obj:`Line`
            The `Line` object of the plot.

        first_run : bool
            Flag signifying whether this is the first run of the `Recorder`. Flipped to `False` after first iteration.
    """

    def __init__(self):
        # type: (Driver) -> None
        """Initialize the `SimpleObjectivePlotter`."""
        super(SimpleObjectivePlotter, self).__init__()

        self.obj_name = None
        self.obj_init = None

        self.xdata = None
        self.ydata = None

        self.ax = None
        self.line = None

        self.first_run = True

    def startup(self, object_requesting_recording):
        # type: (Driver) -> None
        """Obtain the name of the objective function variable from the `Driver` before calling the `super()`.

        Parameters
        ----------
            object_requesting_recording : :obj:`Driver`
                `Driver` that owns this `Recorder`.
        """
        if not isinstance(object_requesting_recording, Driver):
            raise ValueError('This Recorder must be attached to a Driver.')

        super(SimpleObjectivePlotter, self).startup(object_requesting_recording)

    def init_fig(self, fig):
        # type: (Figure) -> None
        """Initialize the axes and line of the plot.

        Parameters
        ----------
            fig : :obj:`Figure`
                Instance of the `Figure` which should be populated.
        """
        self.ax = fig.add_subplot(111)
        self.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        self.ax.set_xlabel('Iteration #')
        self.ax.set_ylabel('Normalized objective function value')

        self.line, = self.ax.plot([], [], color='black')

    def _update_plot(self, *args):
        # type: (dict, dict, dict, dict, dict) -> None
        """Insert the new data into the plot and refresh it.

        Parameters
        ----------
            desvars, responses, objectives, constraints, metadata : dict
                Dictionaries of the new design, response, objective, and constraint variables, as well as metadata.
        """
        if len(args) != 5 and all([isinstance(arg, dict) for arg in args]):
            raise ValueError('Illegal arguments for method _update_plot() of %s' % self.__name__)
        _, _, objectives, _, _ = args

        if self.first_run:
            self.first_run = False
            self.obj_init = objectives.values()[0]
            self.xdata = np.array([0.])
            self.ydata = np.array([1.])
        else:
            _iter = self.xdata[-1] + 1
            self.xdata = np.append(self.xdata, [_iter])
            self.ydata = np.append(self.ydata, [objectives.values()[0]/self.obj_init])
            self.ax.set_xlim([0, _iter])
            self.ax.set_ylim([0, 1])

        self.line.set_data(self.xdata, self.ydata)
        self.ax.relim()
        self.ax.autoscale_view()


class BaseLanePlotter(BaseIterationPlotter):
    """Specialized `BaseIterationPlotter` wrapping a ``lane plot`` style visualization of variables.

    Abstract base class enabling OpenMDAO data to be visualized using colored, horizontal lanes. Each variable to be
    visualized this way has its own lane. The x-axis corresponds to the number of iterations/function evaluations. A
    colorbar is used to indicate the value of a design variable.

    Attributes
    ----------
        n_vars : int
            The number variables.

        var_names : :obj:`list` of :obj:`str`
            List of all variable names.

        xs, ys, cs: :obj:`np.ndarray`
            Arrays containing the x-, y-, and color data of the figure.

        iter : int
            Number of the last iteration.

        ax : :obj:`Axes`
            Matplotlib `Axes` of the plot.

        max_iter : int
            Maximum number of iterations.

        quad : :obj:`matplotlib.collections.QuadMesh`
            Instance of `QuadMesh` that represents the actual plot.

        vmin, vmax : float
            Lower and upper cutoff for  values along the colorbar.

        cmap : str
            Name of the colormap to use.

        norm : :obj:`colors.Normalize`, optional
            Which normalization scheme to use for the colorbar.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, vmin=0., vmax=1., cmap='viridis', norm=None):
        # type: (float, float, str, Optional[colors.Normalize]) -> None
        """Initialize a new `BaseLanePlotter` instance.

        Parameters
        ----------
            vmin, vmax : float
                Lower and upper cutoff for the values along the colorbar.

            cmap : str('viridis')
                Name of the colormap to use for the plot.

            norm : :obj:`colors.Normalize`, optional
                Instance of `colors.Normalize` can be supplied to use a normalization scheme for the colorbar.
        """
        super(BaseLanePlotter, self).__init__()

        self.n_vars = None
        self.var_names = None

        self.xs = None
        self.ys = None
        self.cs = None

        self.iter = 0
        self.ax = None
        self.max_iter = 1000

        self.quad = None

        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap
        self.norm = norm

    def startup(self, object_requesting_recording):
        # type: (Driver) -> None
        """Make sure this `Recorder` is attached to a `Driver` and obtain the maximum number of iterations.

        Parameters
        ----------
            object_requesting_recording : :obj:`Driver`
                Instance of `Driver` to which this `Recorder` is attached.
        """
        if not isinstance(object_requesting_recording, Driver):
            raise ValueError('This Recorder should be attached to a Driver.')

        if 'maxiter' in object_requesting_recording.options:
            self.max_iter = object_requesting_recording.options['maxiter']

        super(BaseLanePlotter, self).startup(object_requesting_recording)

    @abstractmethod
    def init_vars(self):
        # type: () -> None
        """Initialize the variables of the plot.

        This method should be implemented by subclasses such that they can control how variables are initialized.
        """
        raise NotImplementedError

    def init_fig(self, fig):
        # type: (Figure) -> None
        """Initialize the figure, setting up axes, labels, the colorbar, etc.

        Parameters
        ----------
            fig : :obj:`Figure`
                Instance of the `Figure` which should be populated.
        """
        self.init_vars()

        self.xs, self.ys = np.meshgrid(np.arange(0., self.max_iter+.5)-.5, np.arange(0., self.n_vars+.5)-.5)
        self.cs = np.zeros((self.n_vars, self.max_iter))

        self.ax = fig.add_subplot(111)
        self.ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        self.ax.yaxis.set_ticks(np.arange(0, self.n_vars))
        self.ax.yaxis.set_ticklabels(self.var_names)

        self.ax.set_xlim([-.5, .5])
        self.ax.set_ylim([-.5, self.n_vars-.5])
        self.quad = self.ax.pcolormesh(self.xs, self.ys, self.cs,
                                       vmin=self.vmin, vmax=self.vmax, cmap=self.cmap, norm=self.norm)

        fig.colorbar(self.quad)

        self.ax.set_xlabel('Evaluation #')

    @abstractmethod
    def _compute_new_data(self, desvars, responses, objectives, constraints, metadata):
        # type: (dict, dict, dict, dict, dict) -> np.ndarray
        """Return a 1D numpy.ndarray containing the new data points.

        Parameters
        ----------
            desvars, responses, objectives, constraints, metadata : dict
                Dictionaries of the new design, response, objective, and constraint variables, as well as metadata.

        Returns
        -------
            np.ndarray
                A 1D numpy array containing the new data.
        """
        raise NotImplementedError

    def _update_plot(self, *args):
        # type: (dict, dict, dict, dict, dict) -> None
        """Insert the new data into the plot and refresh it.

        Parameters
        ----------
            desvars, responses, objectives, constraints, metadata : dict
                Dictionaries of the new design, response, objective, and constraint variables, as well as metadata.
        """
        data = self._compute_new_data(*args)
        self.cs[:, self.iter] = data[:]
        self.quad.set_array(self.cs.ravel())
        self.ax.set_xlim([-.5, self.iter+.5])
        self.iter += 1


class NormalizedDesignVarPlotter(BaseLanePlotter):
    """Specific case of the `BaseLanePlotter` which plots all normalized design variables of a `Problem`.

    Design variable values are normalized using the ``ref0`` and ``ref`` properties of the design variables as
    specified in the ``metadata``.
    """

    def __init__(self, vmin=0., vmax=1., cmap='viridis', norm=None):
        # type: (float, float, str, Optional[colors.Normalize]) -> None
        """Initialize the `BaseLanePlotter` and stores the ``desvar_metadata`` from the `Driver`.

        Parameters
        ----------
            vmin, vmax : float
                Lower and upper cutoff for the values along the colorbar.

            cmap : str('viridis')
                Name of the colormap to use for the plot.

            norm : :obj:`colors.Normalize`, optional
                Instance of `colors.Normalize` can be supplied to use a normalization scheme for the colorbar.
        """
        super(NormalizedDesignVarPlotter, self).__init__(vmin, vmax, cmap, norm)
        self.desvar_meta = None

    def startup(self, object_requesting_recording):
        # type: (Driver) -> None
        """Make sure this `Recorder` is attached to a `Driver` and obtain the design variable metadata.

        Parameters
        ----------
            object_requesting_recording : :obj:`Driver`
                Instance of `Driver` to which this `Recorder` is attached.
        """
        self.desvar_meta = object_requesting_recording._designvars.copy()
        super(NormalizedDesignVarPlotter, self).startup(object_requesting_recording)

    def init_vars(self):
        # type: () -> None
        """Initialize the lists of design variable names and obtain the number of design variables."""
        self.var_names = list()
        self.n_vars = 0
        for key in self.desvar_meta.keys():
            ref0 = self.desvar_meta[key]['ref0']
            if isinstance(ref0, np.ndarray):
                size = self.desvar_meta[key]['adder'].size
            else:
                size = 1
            self.n_vars += size
            self.var_names.extend(['%s[%d]' % (key, i) for i in range(size)])

    def _compute_new_data(self, desvars, responses, objectives, constraints, metadata):
        # type: (dict, dict, dict, dict, dict) -> np.ndarray
        """Compute the new data points of the plot from the design variable values.

        Parameters
        ----------
            desvars, responses, objectives, constraints, metadata : dict
                Dictionaries of the new design, response, objective, and constraint variables, as well as metadata.

        Returns
        -------
            np.ndarray
                A 1D numpy array containing the new data.
        """
        parts = [(desvars[key] + self.desvar_meta[key]['ref0']) /
                 (self.desvar_meta[key]['ref'] - self.desvar_meta[key]['ref0']) for key in self.desvar_meta.keys()]
        for index, part in enumerate(parts):
            if type(part) == np.ndarray:
                parts[index] = part.flatten()
            else:
                parts[index] = np.array([part])
        return np.concatenate(parts)


class ConstraintsPlotter(BaseLanePlotter):
    """Specific case of the `BaseLanePlotter` plotting all the constraint variables of a `Problem.

    A symmetric logarithmic colorbar is used by default by this plottter.

    Attributes
    ----------
        constr_meta : dict
            A copy of the constraint metadata of the `Driver` this `Recorder` is associated with.
    """

    def __init__(self, vmin=-1., vmax=1., cmap='RdBu_r',
                 norm=colors.SymLogNorm(linthresh=1.e-3, linscale=.5, vmin=-1., vmax=1.)):
        # type: (float, float, str, Optional[colors.Normalize]) -> None
        """Initialize the `BaseLanePlotter` and store the ``constraint_metadata`` from the `Driver`.

        Parameters
        ----------
            vmin, vmax : float
                Lower and upper cutoff for the values along the colorbar.

            cmap : str('RdBu_r')
                Name of the colormap to use for the plot.

            norm : :obj:`colors.Normalize`(`colors.SymLogNorm`), optional
                A symmetric logarithmic colorbar is used by default by this plottter.
        """
        super(ConstraintsPlotter, self).__init__(vmin, vmax, cmap, norm)
        self.constr_meta = None

    def startup(self, object_requesting_recording):
        # type: (Driver) -> None
        """Make sure this `Recorder` is attached to a `Driver` and obtain the constraint variable metadata.

        Parameters
        ----------
            object_requesting_recording : :obj:`Driver`
                Instance of `Driver` to which this `Recorder` is attached.
        """
        self.constr_meta = object_requesting_recording._cons.copy()
        super(ConstraintsPlotter, self).startup(object_requesting_recording)

    def init_vars(self):
        # type: () -> None
        """Initialize the list of constraint variable names and obtain the number of constraints."""
        self.var_names = list()
        self.n_vars = 0
        for key in self.constr_meta.keys():
            size = self.constr_meta[key]['size']
            self.n_vars += size
            self.var_names.extend(['%s[%d]' % (key, i) for i in range(size)])

    def _compute_new_data(self, desvars, responses, objectives, constraints, metadata):
        # type: (dict, dict, dict, dict, dict) -> np.ndarray
        """Compute the new data points for the lane plot from the constraints.

        Parameters
        ----------
            desvars, responses, objectives, constraints, metadata : dict
                Dictionaries of the new design, response, objective, and constraint variables, as well as metadata.

        Returns
        -------
            np.ndarray
                A 1D numpy array containing the new data.
        """
        parts = [constraints[key] for key in self.constr_meta.keys()]
        for index, part in enumerate(parts):
            if type(part) == np.ndarray:
                parts[index] = part.flatten()
            else:
                parts[index] = np.array([part])
        return np.concatenate(parts)
