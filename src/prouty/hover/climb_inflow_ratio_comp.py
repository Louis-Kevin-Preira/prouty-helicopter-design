"""
ClimbInflowRatioComp -- local inflow ratio of a blade element in vertical flight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 2, "Conditions at the Blade Element" p. 96 (Figure 2.2); the hover
form is Chapter 1, step 5, p. 70 (InflowRatioComp).

Momentum with the climb velocity, dT = 4 pi rho r (V_c + v1) v1 dr, equated to
the blade element, dT = 0.5 rho (Omega r)^2 b c a (theta - (V_c + v1)/(Omega r)) dr,
gives p. 96:

         -(Omega a c b/2 + 4 pi V_c) + sqrt[(Omega a c b/2 + 4 pi V_c)^2 + 8 pi b Omega^2 a c r (theta - V_c/(Omega r))]
    v1 = ----------------------------------------------------------------------------------------------------------
                                                         8 pi

Divided by Omega r, with P = a b (c/R) / (16 pi r/R) (as in step 5) and the
local climb ratio lam = (V_c/Omega R) / (r/R):

    vi_Or = [-(lam + 2P) + sqrt((lam + 2P)^2 + 8P (theta - lam))] / 2

which is the step 5 form P [-1 + sqrt(1 + 2 theta/P)] at V_c = 0.

The output v1_Or is the total inflow ratio (V_c + v1)/(Omega r) = lam + vi_Or,
the tangent of the inflow angle of Figure 2.2. Downstream, AngleOfAttackComp
and InducedTorqueLoadingComp then give the angle of attack and the induced plus
climb ("inflow drag", p. 95) torque without change; in hover v1_Or is the step 5
inflow ratio. vi_Or is the induced part alone. V_c < 0 is descent.

The radicand is (lam - 2P)^2 + 8 P theta: it stays positive for any climb or
descent rate as long as the pitch is positive, so the equation always returns
an answer. It is only physical for climb and low rates of descent (momentum
breaks down at about v_1hov/4, p. 95): beyond that, use G0/G5 to check the
flow state. A negative radicand (negative pitch) is floored with a warning, as
in step 5.

    a, b, c_R, r_R, theta (nn,), V_c, V_tip --> v1_Or, vi_Or (nn,)
"""

import warnings

import numpy as np
import openmdao.api as om

DEG = np.pi / 180.0
TINY = 1e-12


class ClimbInflowRatioComp(om.ExplicitComponent):
    """Inflow ratio at each blade station with a climb velocity, p. 96."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=11)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, int)
        self._warned = False

        self.add_input('a', shape=(nn,), units='1/deg', desc='lift curve slope')
        self.add_input('b', val=4.0, desc='number of blades')
        self.add_input('c_R', shape=(nn,), desc='local chord ratio, c/R')
        self.add_input('r_R', shape=(nn,), desc='radial stations, r/R')
        self.add_input('theta', shape=(nn,), units='deg', desc='local pitch')
        self.add_input('V_c', val=0.0, units='ft/s', desc='rate of climb, < 0 in descent')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed, Omega R')

        self.add_output('v1_Or', shape=(nn,), desc='total inflow ratio (V_c + v1) / (Omega r)')
        self.add_output('vi_Or', shape=(nn,), desc='induced inflow ratio v1 / (Omega r)')

        self.declare_partials(['v1_Or', 'vi_Or'], ['a', 'c_R', 'r_R', 'theta'], rows=ar, cols=ar)
        self.declare_partials(['v1_Or', 'vi_Or'], ['b', 'V_c', 'V_tip'], rows=ar, cols=zeros)

    def _state(self, inputs):
        a = inputs['a'] / DEG
        th = inputs['theta'] * DEG
        P = a * inputs['b'] * inputs['c_R'] / (16.0 * np.pi * inputs['r_R'])
        lam = inputs['V_c'] / inputs['V_tip'] / inputs['r_R']
        B = lam + 2.0 * P
        rad = B ** 2 + 8.0 * P * (th - lam)
        if not self._warned and np.any(np.real(rad) <= 0.0):
            self._warned = True
            warnings.warn('the climb inflow radicand went negative; the closed form of '
                          'p. 96 assumes a positive pitch at every station.')
        rad = np.where(np.real(rad) > TINY, rad, TINY)
        return P, th, lam, B, np.sqrt(rad)

    def compute(self, inputs, outputs):
        P, th, lam, B, S = self._state(inputs)
        vi = 0.5 * (S - B)
        outputs['vi_Or'] = vi
        outputs['v1_Or'] = lam + vi

    def compute_partials(self, inputs, partials):
        P, th, lam, B, S = self._state(inputs)
        # vi = (S - B)/2, S^2 = B^2 + 8 P (th - lam), B = lam + 2P
        dvi_dP = 0.5 * ((B * 2.0 + 4.0 * (th - lam)) / S - 2.0)
        dvi_dth = 2.0 * P / S
        dvi_dlam = 0.5 * ((B - 4.0 * P) / S - 1.0)

        r_R, V_c, V_tip = inputs['r_R'], inputs['V_c'], inputs['V_tip']
        dP = {'a': P / inputs['a'], 'b': P / inputs['b'], 'c_R': P / inputs['c_R'],
              'r_R': -P / r_R}
        dlam = {'r_R': -lam / r_R, 'V_c': 1.0 / (V_tip * r_R) + 0.0 * lam,
                'V_tip': -lam / V_tip}

        for name in ('a', 'b', 'c_R', 'r_R', 'theta', 'V_c', 'V_tip'):
            d_vi = (dvi_dP * dP.get(name, 0.0) + dvi_dlam * dlam.get(name, 0.0)
                    + (dvi_dth * DEG if name == 'theta' else 0.0))
            partials['vi_Or', name] = d_vi
            partials['v1_Or', name] = d_vi + dlam.get(name, 0.0)
