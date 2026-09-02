"""
Stub airfoil model on the rotor disc, standing in for Chapter 6.

NOT a physical model. It pins the interface NumericalRotorGroup expects:

    inputs   alpha (rad), M, sec_Lambda, d_alpha_stall   on the field shape
    outputs  cl_raw, cd                                  on the field shape

The lift is linear, c_l = a alpha, with no stall at all -- which is exactly
why the p. 221 bounds matter: without them a linear model hands back c_l = 21
at the 200 deg angles of attack that occur inside the reverse flow circle.
sec_Lambda and d_alpha_stall are accepted and ignored, which is the behaviour
a model without an external stall angle should have.

Replace with AirfoilForwardFlightGroup; do not carry these numbers forward.
"""

import numpy as np
import openmdao.api as om


class DiscAirfoilStub(om.ExplicitComponent):
    """Linear lift, constant drag, over the whole disc."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('num_azimuth', types=int, default=24)
        self.options.declare('num_radial', types=int, default=21)
        self.options.declare('a', types=float, default=6.0)
        self.options.declare('cd', types=float, default=0.0100)

    def setup(self):
        shape = (self.options['num_nodes'], self.options['num_azimuth'],
                 self.options['num_radial'])
        rows = np.arange(int(np.prod(shape)))

        self.add_input('alpha', shape=shape, units='rad')
        self.add_input('M', shape=shape)
        self.add_input('sec_Lambda', shape=shape, val=1.0)
        self.add_input('d_alpha_stall', shape=shape, val=0.0, units='rad')

        self.add_output('cl_raw', shape=shape)
        self.add_output('cd', shape=shape)

        self.declare_partials('cl_raw', 'alpha', rows=rows, cols=rows,
                              val=self.options['a'])

    def compute(self, inputs, outputs):
        outputs['cl_raw'] = self.options['a'] * inputs['alpha']
        outputs['cd'] = np.full(inputs['alpha'].shape, self.options['cd'])
