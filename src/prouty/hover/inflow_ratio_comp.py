"""
InflowRatioComp -- local inflow ratio v1 / (Omega r).

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 1, step 5, p. 70.

    v1        a b (c/R)   [          sqrt(1 + 32 pi theta (r/R) / (a b c/R)) ]
    ----- = ------------- x [ -1  +                                           ]
    Omega r  16 pi (r/R)    [                                                 ]

This is the closed form solution of the combined theories at one station:
momentum gives dT = 4 pi rho r v1^2 dr, the blade element gives
dT = 0.5 rho (Omega r)^2 b c a (theta - phi) dr, and equating them leaves the
quadratic phi^2 + K phi - K theta = 0 with K = a b (c/R) / (8 pi r/R). The
positive root is the equation above. Setting sigma = b c / (pi R) recovers the
constant-chord form also given on p. 70.

Units. The book notes on p. 70 that "theta and a are in radian units in this
equation", while the airfoil model of Chapter 6 produces a per degree and the
pitch is carried in degrees. Both are converted here, so the interface stays
consistent with LiftModelCoefsComp and PitchComp.

    a, b, c_R, r_R, theta --> InflowRatioComp --> v1_Or (nn,)
"""

import warnings

import numpy as np
import openmdao.api as om

DEG = np.pi / 180.0
TINY = 1e-12


class InflowRatioComp(om.ExplicitComponent):
    """Inflow ratio at each blade station, combined momentum / blade element."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)

    def setup(self):
        nn = self.options['num_nodes']
        self._warned = False

        self.add_input('a', shape=(nn,), units='1/deg', desc='lift curve slope')
        self.add_input('b', val=4.0, desc='number of blades')
        self.add_input('c_R', shape=(nn,), desc='local chord ratio, c/R')
        self.add_input('r_R', shape=(nn,), desc='radial stations, r/R')
        self.add_input('theta', shape=(nn,), units='deg', desc='local pitch')

        self.add_output('v1_Or', shape=(nn,), desc='inflow ratio v1 / (Omega r)')

        ar = np.arange(nn)
        self.declare_partials('v1_Or', ['a', 'c_R', 'r_R', 'theta'],
                              rows=ar, cols=ar)
        self.declare_partials('v1_Or', 'b', rows=ar, cols=np.zeros(nn, int))

    def _state(self, inputs):
        """Prefactor P, radicand and its square root S."""
        a = inputs['a'] / DEG                       # 1/deg -> 1/rad
        th = inputs['theta'] * DEG                  # deg -> rad

        P = a * inputs['b'] * inputs['c_R'] / (16.0 * np.pi * inputs['r_R'])
        rad = 1.0 + 2.0 * th / P

        if not self._warned and np.any(np.real(rad) <= 0.0):
            self._warned = True
            warnings.warn('the inflow radicand went negative; the closed form of '
                          'p. 70 assumes a positive pitch at every station.')
        rad = np.where(np.real(rad) > TINY, rad, TINY)
        return P, th, rad, np.sqrt(rad)

    def compute(self, inputs, outputs):
        P, _, _, S = self._state(inputs)
        outputs['v1_Or'] = P * (S - 1.0)

    def compute_partials(self, inputs, partials):
        P, th, _, S = self._state(inputs)

        # d phi / d P at fixed theta, and d phi / d theta at fixed P.
        dphi_dP = (S - 1.0) - th / (P * S)
        dphi_dth = 1.0 / S

        # P is a product, so its own partials are P divided by each factor.
        partials['v1_Or', 'a'] = dphi_dP * P / inputs['a']
        partials['v1_Or', 'b'] = dphi_dP * P / inputs['b']
        partials['v1_Or', 'c_R'] = dphi_dP * P / inputs['c_R']
        partials['v1_Or', 'r_R'] = -dphi_dP * P / inputs['r_R']
        partials['v1_Or', 'theta'] = dphi_dth * DEG
