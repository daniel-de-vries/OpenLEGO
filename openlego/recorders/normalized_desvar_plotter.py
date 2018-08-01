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

This file contains the definition of the `NormalizedDesignVarPlotter` class.
"""
from __future__ import absolute_import, division, print_function

import matplotlib.colors as colors
import numpy as np
from openmdao.core.driver import Driver
from typing import Optional

from .base_lane_plotter import BaseLanePlotter


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
                size = ref0.size
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
        parts = [(desvars[key] - self.desvar_meta[key]['lower']) /
                 (self.desvar_meta[key]['upper'] - self.desvar_meta[key]['lower']) for key in self.desvar_meta.keys()]
        for index, part in enumerate(parts):
            parts[index] = np.atleast_1d(part).flatten()
        return np.concatenate(parts)
