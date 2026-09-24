"""
FlarePitchRateComp -- G2e, maximum nose-down pitch rate at the end of the flare.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Minimum Touchdown Speed" p. 361 (derivation in Chapter 7).

    theta_dot_max = gamma Omega Delta_B1 / 16

Delta_B1: forward cyclic pitch available. Example: 88 deg/s for 8 deg (p. 362).

    gamma (nn,), Omega, delta_B1 (nn,) --> theta_dot_max (nn,)
"""

import numpy as np
import openmdao.api as om


class FlarePitchRateComp(om.ExplicitComponent):
    """Pitch rate the longitudinal control power allows, p. 361."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('gamma', val=8.1 * np.ones(nn), desc='Lock number')
        self.add_input('Omega', val=21.67, units='rad/s')
        self.add_input('delta_B1', val=np.deg2rad(8.0) * np.ones(nn), units='rad')
        self.add_output('theta_dot_max', val=np.ones(nn), units='rad/s')
        self.declare_partials('theta_dot_max', ['gamma', 'delta_B1'], rows=ar, cols=ar)
        self.declare_partials('theta_dot_max', 'Omega', rows=ar, cols=np.zeros(nn, dtype=int))

    def compute(self, inputs, outputs):
        outputs['theta_dot_max'] = inputs['gamma'] * inputs['Omega'] * inputs['delta_B1'] / 16.0

    def compute_partials(self, inputs, J):
        g, w, dB = inputs['gamma'], inputs['Omega'], inputs['delta_B1']
        J['theta_dot_max', 'gamma'] = w * dB / 16.0
        J['theta_dot_max', 'delta_B1'] = g * w / 16.0
        J['theta_dot_max', 'Omega'] = g * dB / 16.0
