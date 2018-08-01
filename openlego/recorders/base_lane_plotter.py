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

This file contains the definition of the `BaseLanePlotter` class.
"""
from __future__ import absolute_import, division, print_function

import abc
from abc import abstractmethod

import matplotlib.colors as colors
import numpy as np
from matplotlib import ticker as ticker
from matplotlib.figure import Figure
from openmdao.core.driver import Driver
from typing import Optional

from .base_iteration_plotter import BaseIterationPlotter


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
        if len(args) != 5 and not any([isinstance(arg, dict) for arg in args]):
            raise ValueError('Illegal arguments for _update_plot of %s' % self.__name__)
        desvars, responses, objectives, constraints, metadata = args

        data = self._compute_new_data(desvars, responses, objectives, constraints, metadata)
        self.cs[:, self.iter] = data[:]
        self.quad.set_array(self.cs.ravel())
        self.ax.set_xlim([-.5, self.iter+.5])
        self.iter += 1
