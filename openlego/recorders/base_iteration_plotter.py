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

This file contains the definitions of the `BaseIterationPlotter` class.
"""
from __future__ import absolute_import, division, print_function

import sys

if sys.version_info[0] == 3:
    import tkinter as tk
else:
    import Tkinter as Tk
    tk = Tk

import abc
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
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from openlego.utils.general_utils import try_hard


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
        # TODO: why is this code commented out?
        #self.options.declare('save_on_close', False, desc='Set to True to save figure when closing the Recorder')
        self.save_settings = {'path': self.__class__.__name__ + '_figure.png',
                              'dpi': 600, 'ar': 1.61803398875, 'width': 4000}

        self._in, self._out = Pipe()
        self._callback_in, self._callback_out = Pipe()
        self._process = None

        self._tk = None
        self._canvas = None
        self._toolbar = None
        self._fig = None

        # TODO: why did this have to change from .options to .recording_options?
        self.recording_options['record_objectives'] = True
        self.recording_options['record_constraints'] = True

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

    def record_iteration_driver_passing_vars(self, object_requesting_recording, desvars, responses, objectives,
                                             constraints, sysvars, metadata):
        super(BaseIterationPlotter, self).record_iteration_driver_passing_vars(object_requesting_recording,
                                                                               desvars, responses, objectives,
                                                                               constraints, sysvars, metadata)
        self._record_iteration_driver(metadata)

    def record_iteration_driver(self, object_requesting_recording, metadata):
        super(BaseIterationPlotter, self).record_iteration_driver(object_requesting_recording, metadata)
        self._record_iteration_driver(metadata)

    def record_iteration_system(self, object_requesting_recording, metadata):
        super(BaseIterationPlotter, self).record_iteration_system(object_requesting_recording, metadata)
        self._record_iteration_system(metadata)

    def record_iteration_solver(self, object_requesting_recording, metadata, **kwargs):
        super(BaseIterationPlotter, self).record_iteration_solver(object_requesting_recording, metadata, **kwargs)
        self._record_iteration_solver(metadata)

    def _record_iteration_driver(self, metadata):
        self._in.send(('update',
                       self._desvars_values,
                       self._responses_values,
                       self._objectives_values,
                       self._constraints_values,
                       metadata))
        self.save_figure()

    def _record_iteration_system(self, metadata):
        self._in.send(('update',
                       self._inputs,
                       self._outputs,
                       self._resids,
                       metadata))
        self.save_figure()

    def _record_iteration_solver(self, metadata):
        self._in.send(('update',
                       self._abs_error,
                       self._rel_error,
                       self._outputs,
                       self._resids,
                       metadata))
        self.save_figure()

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
