"""
TouchdownSpeedComp -- G2e, minimum touchdown speed.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Minimum Touchdown Speed" p. 361.

    V_TD = mu_auto (Omega R) - (g/2) tan(alpha_TPP) Delta_t

(g/2) tan(alpha_TPP) Delta_t: speed lost during the nose-down rotation.
Example: 21 kt (p. 362).

    mu_auto (nn,), V_tip, alpha_TPP (nn,), dt (nn,) --> V_TD (nn,)
"""

import numpy as np
import openmdao.api as om

G = 32.2       # ft/s^2, as printed in Chapter 5


class TouchdownSpeedComp(om.ExplicitComponent):
    """Minimum touchdown speed after the flare, p. 361."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('mu_auto', val=0.09 * np.ones(nn))
        self.add_input('V_tip', val=650.0, units='ft/s')
        self.add_input('alpha_TPP', val=np.deg2rad(45.0) * np.ones(nn), units='rad')
        self.add_input('dt', val=np.ones(nn), units='s')
        self.add_output('V_TD', val=np.ones(nn), units='ft/s')
        self.declare_partials('V_TD', ['mu_auto', 'alpha_TPP', 'dt'], rows=ar, cols=ar)
        self.declare_partials('V_TD', 'V_tip', rows=ar, cols=np.zeros(nn, dtype=int))

    def compute(self, inputs, outputs):
        outputs['V_TD'] = (inputs['mu_auto'] * inputs['V_tip']
                           - 0.5 * G * np.tan(inputs['alpha_TPP']) * inputs['dt'])

    def compute_partials(self, inputs, J):
        a, t = inputs['alpha_TPP'], inputs['dt']
        J['V_TD', 'mu_auto'] = inputs['V_tip'] * np.ones_like(a)
        J['V_TD', 'V_tip'] = inputs['mu_auto']
        J['V_TD', 'alpha_TPP'] = -0.5 * G * t / np.cos(a) ** 2
        J['V_TD', 'dt'] = -0.5 * G * np.tan(a)
