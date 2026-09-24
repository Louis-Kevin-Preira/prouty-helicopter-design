"""
SmoothMinComp -- G3/G4, smooth minimum of two capability limits.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, pp. 364-366: acceleration is limited by the tilted hover thrust
at low speed and by excess power above; deceleration by the acceleration
capability near hover and by rotor autorotation above about 37 kt.

    c = (a + b - sqrt((a - b)^2 + eps^2)) / 2      eps = 0.5 ft/s^2

    a (nn,), b (nn,) --> c (nn,)       names set by options
"""

import numpy as np
import openmdao.api as om

EPS = 0.5      # ft/s^2


class SmoothMinComp(om.ExplicitComponent):
    """Differentiable min of two accelerations."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('a', default='acc_hover')
        self.options.declare('b', default='acc_power')
        self.options.declare('out', default='acc_max')
        self.options.declare('a_scalar', types=bool, default=False,
                             desc='a is one value for all nodes')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        a, b, out = self.options['a'], self.options['b'], self.options['out']
        scalar = self.options['a_scalar']
        self.add_input(a, val=np.ones(1 if scalar else nn), units='ft/s**2')
        self.add_input(b, val=np.ones(nn), units='ft/s**2')
        self.add_output(out, val=np.ones(nn), units='ft/s**2')
        self.declare_partials(out, a, rows=ar, cols=np.zeros(nn, int) if scalar else ar)
        self.declare_partials(out, b, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        a, b = inputs[self.options['a']], inputs[self.options['b']]
        outputs[self.options['out']] = 0.5 * (a + b - np.sqrt((a - b) ** 2 + EPS ** 2))

    def compute_partials(self, inputs, J):
        o = self.options
        a, b = inputs[o['a']], inputs[o['b']]
        s = (a - b) / np.sqrt((a - b) ** 2 + EPS ** 2)
        J[o['out'], o['a']] = 0.5 * (1.0 - s)
        J[o['out'], o['b']] = 0.5 * (1.0 + s)
