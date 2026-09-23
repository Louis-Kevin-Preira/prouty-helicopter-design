"""
LatFlappingComp -- lateral flapping and cyclic.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 169 and p. 200 (tip loss and root cutout).

    A_1 - b_1s = - [ (4/3) mu a_0 p3 + (v1/Omega R) p4 ]
                 / [ p4 + (1/2) mu^2 p2 ]            p_n = B^n - x_0^n

obtained by setting the rotor pitching moment to zero (p. 169). At B = 1,
x_0 = 0 it collapses to the p. 169 form,
- [(4/3) mu a_0 + v1/Omega R] / (1 + mu^2/2).

The two contributions are physically distinct and they add:

  * (4/3) mu a_0 is the lateral moment produced by coning. A coned rotor in
    forward flight sees a larger angle of attack on the front of the disc than
    on the rear, which flaps the disc sideways. It vanishes in hover.
  * v1/Omega R is the uniform induced velocity term. It does not vanish in
    hover, which is why A_1 - b_1s stays finite and non-zero there.

Sign: negative in every level flight case of this chapter, around -2.3 deg for
the example helicopter, and remarkably insensitive to flight condition -- the
Table 3.3 spread over level flight, a 1000 ft/min climb and autorotation is
only 0.2 deg, because a_0 and v1/Omega R barely move.

As with the longitudinal case, only the combination A_1 - b_1s is set by the
aerodynamics. Splitting it needs a rolling moment balance -- tail rotor thrust,
lateral centre of gravity offset -- which is Chapter 8. Prouty notes b_1s can
be taken as negligible for most steady flight (p. 169), so A_1 here IS the
difference.

Note the sign asymmetry with the longitudinal case: it is B_1 + a_1s but
A_1 - b_1s, a consequence of the pitch convention theta = theta_0 +
(r/R) theta_1 - A_1 cos psi - B_1 sin psi of p. 165.

    mu, a0, vi_OR, B, x_0 --> LatFlappingComp --> A1_b1s (nn,)
"""

import numpy as np
import openmdao.api as om

from .collective_pitch_comp import _powers


class LatFlappingComp(om.ExplicitComponent):
    """Lateral flapping A_1 - b_1s, closed form."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('a0', shape=(nn,), units='rad', desc='coning angle')
        self.add_input('vi_OR', shape=(nn,), desc='v1 / (Omega R)')
        self.add_input('B', shape=(nn,), val=1.0, desc='tip loss factor')
        self.add_input('x_0', val=0.0, desc='root cutout r/R')

        self.add_output('A1_b1s', shape=(nn,), units='rad', desc='A_1 - b_1s')

        for name in ('mu', 'a0', 'vi_OR', 'B'):
            self.declare_partials('A1_b1s', name, rows=ar, cols=ar)
        self.declare_partials('A1_b1s', 'x_0', rows=ar,
                              cols=np.zeros(nn, dtype=int))

    def _terms(self, inputs):
        mu = inputs['mu']
        (p1, p2, p3, p4), _, _ = _powers(inputs['B'], inputs['x_0'][0])
        num = (4.0 / 3.0) * mu * inputs['a0'] * p3 + inputs['vi_OR'] * p4
        return num, p2, p3, p4, p4 + 0.5 * mu ** 2 * p2

    def compute(self, inputs, outputs):
        num, _, _, _, den = self._terms(inputs)
        outputs['A1_b1s'] = -num / den

    def compute_partials(self, inputs, partials):
        mu, a0 = inputs['mu'], inputs['a0']
        num, p2, p3, p4, den = self._terms(inputs)
        value = -num / den

        partials['A1_b1s', 'a0'] = -(4.0 / 3.0) * mu * p3 / den
        partials['A1_b1s', 'vi_OR'] = -p4 / den
        partials['A1_b1s', 'mu'] = (-(4.0 / 3.0) * a0 * p3
                                    - value * mu * p2) / den

        _, dB, dx = _powers(inputs['B'], inputs['x_0'][0])
        for name, (d1, d2, d3, d4) in (('B', dB), ('x_0', dx)):
            d_num = (4.0 / 3.0) * mu * a0 * d3 + inputs['vi_OR'] * d4
            d_den = d4 + 0.5 * mu ** 2 * d2
            partials['A1_b1s', name] = (-d_num - value * d_den) / den
