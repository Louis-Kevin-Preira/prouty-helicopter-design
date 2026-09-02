"""
Stub standing in for the Chapter 6 airfoil model.

NOT a physical model. It exists only so that MeanDragCoefGroup can be wired
and tested before prouty.airfoil is available, and to pin down the interface
the real group must expose:

    input   'alpha'  units='rad'
    input   'M'
    output  'cd'

The drag law is the classical three-term polar of p. 205,

    c_d = c_d0 + c_d1 alpha + c_d2 alpha^2      alpha in radians

with the Bailey constants that Prouty plots against 0012 data in Figure 1.37,
plus a crude Prandtl-Glauert-like Mach factor so that the M input is not
ignored. Replace the whole subsystem with AirfoilHoverGroup; do not carry
these numbers forward.
"""

import numpy as np
import openmdao.api as om

CD0, CD1, CD2 = 0.0087, -0.0216, 0.400          # Figure 1.37, p. 205-206


class _Naca0012DragStub(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('alpha', shape=(nn,), units='rad')
        self.add_input('M', shape=(nn,))
        self.add_output('cd', shape=(nn,))

        self.declare_partials('cd', ['alpha', 'M'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        alpha, M = inputs['alpha'], inputs['M']
        outputs['cd'] = (CD0 + CD1 * alpha + CD2 * alpha ** 2) \
            / np.sqrt(1.0 - M ** 2)

    def compute_partials(self, inputs, partials):
        alpha, M = inputs['alpha'], inputs['M']
        pg = np.sqrt(1.0 - M ** 2)
        polar = CD0 + CD1 * alpha + CD2 * alpha ** 2

        partials['cd', 'alpha'] = (CD1 + 2.0 * CD2 * alpha) / pg
        partials['cd', 'M'] = polar * M / pg ** 3


class AirfoilStubGroup(om.Group):
    """Minimal group with the interface MeanDragCoefGroup expects."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        self.add_subsystem(
            'drag', _Naca0012DragStub(num_nodes=self.options['num_nodes']),
            promotes=['*'])
