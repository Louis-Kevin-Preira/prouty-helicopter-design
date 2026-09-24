"""
FlareAngleComp -- G2e, maximum allowable tip path plane angle at the end of
the cyclic flare.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 5, "Minimum Touchdown Speed" p. 361.

    alpha_TPP = min(theta_dot_max Delta_t, 45 deg)

The min is smoothed over +/- 1 deg so the output stays differentiable.
Example: 110 deg before the limit (printed 100, p. 362), hence 45 deg.

    theta_dot_max (nn,), dt (nn,) --> alpha_raw (nn,), alpha_TPP (nn,)
"""

import numpy as np
import openmdao.api as om

ALPHA_MAX = np.deg2rad(45.0)
SMOOTH = np.deg2rad(1.0)


class FlareAngleComp(om.ExplicitComponent):
    """Flare angle limited by control power, time available and 45 deg, p. 361."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        self.add_input('theta_dot_max', val=np.ones(nn), units='rad/s')
        self.add_input('dt', val=np.ones(nn), units='s')
        self.add_output('alpha_raw', val=np.ones(nn), units='rad')
        self.add_output('alpha_TPP', val=0.5 * np.ones(nn), units='rad')
        self.declare_partials(['alpha_raw', 'alpha_TPP'], ['theta_dot_max', 'dt'],
                              rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        a = inputs['theta_dot_max'] * inputs['dt']
        d = a - ALPHA_MAX
        outputs['alpha_raw'] = a
        outputs['alpha_TPP'] = 0.5 * (a + ALPHA_MAX - np.sqrt(d ** 2 + SMOOTH ** 2))

    def compute_partials(self, inputs, J):
        q, t = inputs['theta_dot_max'], inputs['dt']
        d = q * t - ALPHA_MAX
        s = 0.5 * (1.0 - d / np.sqrt(d ** 2 + SMOOTH ** 2))
        J['alpha_raw', 'theta_dot_max'] = t
        J['alpha_raw', 'dt'] = q
        J['alpha_TPP', 'theta_dot_max'] = s * t
        J['alpha_TPP', 'dt'] = s * q
