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

This file contains the definition of the `ConstraintsPlotter` class.
"""
from __future__ import absolute_import, division, print_function

import matplotlib.colors as colors
import numpy as np
from openmdao.core.driver import Driver
from typing import Optional

from .base_lane_plotter import BaseLanePlotter


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
            parts[index] = np.atleast_1d(part).flatten()
        return np.concatenate(parts)
