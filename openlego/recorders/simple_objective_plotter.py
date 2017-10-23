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

This file contains the definition of the `SimpleObjectivePlotter` class.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import ticker as ticker
from matplotlib.figure import Figure
from openmdao.core.driver import Driver

from .base_iteration_plotter import BaseIterationPlotter


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
