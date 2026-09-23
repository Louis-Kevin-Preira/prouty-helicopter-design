"""
TwistDistComp -- built-in blade twist distribution.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 3 ("local twist, d theta"), p. 69; twist conventions p. 14,
Figure 1.7.

Step 4 of the method, p. 69, adds the collective to this distribution as
theta = theta_0 + d_theta, so what is produced here is the twist increment
alone, not the pitch. Prouty plots linear twist from the centre of rotation
in Figure 1.7, which makes theta_0 the pitch at r/R = 0; setting
reference = 'cutout' instead makes theta_0 the pitch at the root cutout, the
convention used when twist is quoted over the aerodynamic span only.

    d_theta = theta_1 (r/R - x_ref)          [deg]

Ideal twist is deliberately not handled here: theta = theta_t / (r/R), p. 19,
is multiplicative in the collective and so belongs to PitchComp, not to an
additive twist increment.

    r_R, theta_1 --> TwistDistComp --> d_theta (nn,)
"""

import numpy as np
import openmdao.api as om


class TwistDistComp(om.ExplicitComponent):
    """Linear twist increment about the chosen reference station."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)
        self.options.declare('reference', values=('center', 'cutout'),
                             default='center',
                             desc='station where the twist increment is zero')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('r_R', shape=(nn,), desc='radial stations, r/R')
        self.add_input('theta_1', val=-10.0, units='deg',
                       desc='linear twist rate over the full radius')
        self.add_output('d_theta', shape=(nn,), units='deg',
                        desc='twist increment')

        ar = np.arange(nn)
        self.declare_partials('d_theta', 'r_R', rows=ar, cols=ar)
        self.declare_partials('d_theta', 'theta_1', rows=ar, cols=np.zeros(nn, int))

        if self.options['reference'] == 'cutout':
            self.add_input('x0', val=0.15, desc='root cutout, r/R')
            self.declare_partials('d_theta', 'x0', rows=ar, cols=np.zeros(nn, int))

    def _x_ref(self, inputs):
        return inputs['x0'] if self.options['reference'] == 'cutout' else 0.0

    def compute(self, inputs, outputs):
        outputs['d_theta'] = inputs['theta_1'] * (inputs['r_R'] - self._x_ref(inputs))

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        partials['d_theta', 'r_R'] = np.full(nn, inputs['theta_1'])
        partials['d_theta', 'theta_1'] = inputs['r_R'] - self._x_ref(inputs)
        if self.options['reference'] == 'cutout':
            partials['d_theta', 'x0'] = np.full(nn, -inputs['theta_1'])
