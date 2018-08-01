#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright 2018 D. de Vries

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file contains the definitions of the `VOIPlotter` class.
"""
from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np
from matplotlib import ticker as ticker
from matplotlib.figure import Figure
from mpl_toolkits import axisartist as aa
from mpl_toolkits.axes_grid1.parasite_axes import host_subplot_class_factory
from openmdao.core.system import System

from .base_iteration_plotter import BaseIterationPlotter


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
        inputs, outputs, resids, _ = args

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
