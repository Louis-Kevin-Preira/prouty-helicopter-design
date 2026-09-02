"""
InducedVelocityComp -- momentum induced velocity ratio in forward flight.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 167 (high speed form) and p. 122-123 (exact momentum form).

form = 'high_speed'  (p. 167, the form used everywhere in this chapter)

    v1 / Omega R = (C_T/sigma) sigma / (2 mu)  =  C_T / (2 mu)

    It comes from v1 = T / (2 rho A V), the wing analogy of p. 122, and is
    what feeds the inflow ratio of p. 167, the coning of p. 213 and the
    alpha_TPP identity of p. 198. It is singular at mu = 0.

form = 'exact'  (p. 123)

    T = rho A (2 v1) sqrt(V^2 + v1^2)   solved without approximation:

    v1 / Omega R = sqrt{ [sqrt(mu^4 + C_T^2) - mu^2] / 2 }

    Regular everywhere: it tends to sqrt(C_T/2), the hover value, as mu -> 0,
    and to C_T/(2 mu) at high speed, so it can be used as a drop-in
    replacement that removes the singularity.

Prouty states the two agree above roughly 30 knots for the example helicopter
(Figure 3.3, p. 124). In tip speed ratio terms that is mu = 0.08 -- but the
gap is still 5 % at mu = 0.10, which is the lowest chart of this chapter, so
the choice is not purely academic at the bottom of the speed range.

Both forms assume cos(alpha_TPP) = 1, i.e. the free stream lies in the disc
plane. Prouty adopts that convention explicitly on p. 122 and every equation
downstream is written for it.

    CT_sigma, sigma, mu --> InducedVelocityComp --> vi_OR (nn,)
"""

import numpy as np
import openmdao.api as om


class InducedVelocityComp(om.ExplicitComponent):
    """Momentum induced velocity divided by tip speed."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('form', values=('high_speed', 'exact'),
                             default='high_speed',
                             desc="'high_speed': p. 167; 'exact': p. 123")

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('CT_sigma', shape=(nn,), desc='C_T / sigma')
        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('sigma', val=0.085, desc='rotor solidity')

        self.add_output('vi_OR', shape=(nn,), desc='v1 / (Omega R)')

        ar = np.arange(nn)
        for name in ('CT_sigma', 'mu'):
            self.declare_partials('vi_OR', name, rows=ar, cols=ar)
        self.declare_partials('vi_OR', 'sigma', rows=ar,
                              cols=np.zeros(nn, dtype=int))

    def compute(self, inputs, outputs):
        CT = inputs['sigma'][0] * inputs['CT_sigma']
        mu = inputs['mu']

        if self.options['form'] == 'high_speed':
            if np.any(np.real(mu) <= 0.0):
                raise om.AnalysisError(
                    "InducedVelocityComp: the 'high_speed' form is singular at "
                    f"mu = 0, got {np.real(mu)}. Use form='exact' near hover.")
            outputs['vi_OR'] = CT / (2.0 * mu)
        else:
            outputs['vi_OR'] = np.sqrt(
                0.5 * (np.sqrt(mu ** 4 + CT ** 2) - mu ** 2))

    def compute_partials(self, inputs, partials):
        sigma = inputs['sigma'][0]
        CT_sigma, mu = inputs['CT_sigma'], inputs['mu']
        CT = sigma * CT_sigma

        if self.options['form'] == 'high_speed':
            vi = CT / (2.0 * mu)
            partials['vi_OR', 'CT_sigma'] = vi / CT_sigma
            partials['vi_OR', 'sigma'] = vi / sigma
            partials['vi_OR', 'mu'] = -vi / mu
        else:
            w = np.sqrt(mu ** 4 + CT ** 2)
            vi = np.sqrt(0.5 * (w - mu ** 2))
            dvi_dCT = CT / (4.0 * vi * w)

            partials['vi_OR', 'CT_sigma'] = dvi_dCT * sigma
            partials['vi_OR', 'sigma'] = dvi_dCT * CT_sigma
            partials['vi_OR', 'mu'] = (mu ** 3 / w - mu) / (2.0 * vi)
