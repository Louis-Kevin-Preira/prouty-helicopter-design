"""
RotorSpeedDecayComp -- G2a, rotor speed after a power failure.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Rotor Speed Decay" pp. 348-350, Figure 5.4.

Decelerating torque proportional to Omega^2 (conservative, p. 348):

    Omega/Omega_0 = 1 / (1 + f t / (2 t_KE))

f is the fraction of the power lost: 1 for a complete failure,
1/2 for one engine of two when the other keeps its power (17 % in the
first second for the example, p. 350).

    t (nn,), t_KE, f --> Omega_ratio (nn,)
"""

import numpy as np
import openmdao.api as om


class RotorSpeedDecayComp(om.ExplicitComponent):
    """Omega/Omega_0 at the times t after the failure, pp. 349-350."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zero = np.zeros(nn, dtype=int)
        self.add_input('t', val=np.ones(nn), units='s')
        self.add_input('t_KE', val=1.2, units='s')
        self.add_input('power_loss_fraction', val=1.0)
        self.add_output('Omega_ratio', val=np.ones(nn))
        self.declare_partials('Omega_ratio', 't', rows=ar, cols=ar)
        self.declare_partials('Omega_ratio', ['t_KE', 'power_loss_fraction'], rows=ar, cols=zero)

    def compute(self, inputs, outputs):
        s = inputs['power_loss_fraction'] * inputs['t'] / (2.0 * inputs['t_KE'])
        outputs['Omega_ratio'] = 1.0 / (1.0 + s)

    def compute_partials(self, inputs, J):
        t, tk, f = inputs['t'], inputs['t_KE'], inputs['power_loss_fraction']
        d = -1.0 / (1.0 + f * t / (2.0 * tk)) ** 2       # d ratio / d s
        J['Omega_ratio', 't'] = d * f / (2.0 * tk)
        J['Omega_ratio', 't_KE'] = -d * f * t / (2.0 * tk ** 2)
        J['Omega_ratio', 'power_loss_fraction'] = d * t / (2.0 * tk)
