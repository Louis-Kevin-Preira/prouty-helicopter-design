"""
CollectivePitchComp -- closed-form link between collective pitch and thrust.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 167 (C_T/sigma), p. 168 (theta_0), p. 200 (tip loss and root
cutout).

One relation, used in both directions:

  mode = 'collective'   C_T/sigma known, solve for theta_0        p. 168
  mode = 'thrust'       theta_0 known, solve for C_T/sigma        p. 167

Trim needs the first, the isolated rotor charts need the second -- their
curves are labelled in theta_0 (p. 254). Coding one component with two modes
guarantees the two directions stay exact inverses of each other, which two
separate components would not.

The relation is linear in C_T/sigma, theta_0, theta_1 and lambda':

    K_T (C_T/sigma) = D theta_0 + K_1 theta_1 + K_L lambda'

with, writing p_n = B^n - x_0^n (p. 200):

    K_T = (4/a) [p4 + (3/2) mu^2 p2]
    K_1 = (1/2) [p4^2 - (3/2) mu^2 p4 p2 + (3/2) mu^4 p2^2]
    K_L = p2 [p4 - mu^2/2]
    D   = (2/3) p3 p4 - (5/3) mu^2 p3 p2 + mu^2 p1 p4 + (3/2) mu^4 p1 p2

Tip loss and root cutout are carried by B and x_0 as ordinary inputs rather
than by a discrete option. Setting B = 1 and x_0 = 0 collapses the four
coefficients to the p. 167-168 forms exactly -- (4/a)(1 + 1.5 mu^2),
(1/2)(1 - 1.5 mu^2 + 1.5 mu^4), (1 - mu^2/2) and 2/3 - (2/3) mu^2 + 1.5 mu^4
-- so nothing is lost, and the model stays differentiable with respect to B.
That matters because Prouty computes B from the hover equation rather than
fixing it (p. 199), which makes it a function of C_T, not a constant.

Note that theta_0 here is the collective at the blade ROOT, since the pitch
distribution is theta_0 + (r/R) theta_1 (p. 165). The isolated rotor charts
are labelled in the same variable, but Prouty's twist correction of p. 230
converts between twists at fixed 0.75 R pitch -- do not confuse the two.

    mu, CT_sigma, lambda_p, theta_1, a, B, x_0 --> theta_0     ('collective')
    mu, theta_0,  lambda_p, theta_1, a, B, x_0 --> CT_sigma    ('thrust')
"""

import numpy as np
import openmdao.api as om


def _powers(B, x_0):
    """p_n = B^n - x_0^n for n = 1..4, and their derivatives."""
    p = [B ** n - x_0 ** n for n in (1, 2, 3, 4)]
    dB = [n * B ** (n - 1) for n in (1, 2, 3, 4)]
    dx = [-n * x_0 ** (n - 1) for n in (1, 2, 3, 4)]
    return p, dB, dx


def _coefficients(mu, B, x_0, a):
    """K_T, K_1, K_L, D of p. 200."""
    (p1, p2, p3, p4), _, _ = _powers(B, x_0)
    mu2, mu4 = mu ** 2, mu ** 4

    K_T = 4.0 / a * (p4 + 1.5 * mu2 * p2)
    K_1 = 0.5 * (p4 ** 2 - 1.5 * mu2 * p4 * p2 + 1.5 * mu4 * p2 ** 2)
    K_L = p2 * (p4 - 0.5 * mu2)
    D = ((2.0 / 3.0) * p3 * p4 - (5.0 / 3.0) * mu2 * p3 * p2
         + mu2 * p1 * p4 + 1.5 * mu4 * p1 * p2)
    return K_T, K_1, K_L, D


def _d_coefficients(mu, B, x_0, a, wrt):
    """Derivatives of K_T, K_1, K_L, D with respect to 'mu', 'B' or 'x_0'."""
    (p1, p2, p3, p4), dB, dx = _powers(B, x_0)
    mu2, mu4 = mu ** 2, mu ** 4

    if wrt == 'mu':
        return (4.0 / a * 3.0 * mu * p2,
                0.5 * (-3.0 * mu * p4 * p2 + 6.0 * mu ** 3 * p2 ** 2),
                -mu * p2,
                (-(10.0 / 3.0) * mu * p3 * p2 + 2.0 * mu * p1 * p4
                 + 6.0 * mu ** 3 * p1 * p2))

    d1, d2, d3, d4 = dB if wrt == 'B' else dx

    # partials of each coefficient with respect to p1..p4, then chain rule
    K_T = 4.0 / a * (d4 + 1.5 * mu2 * d2)
    K_1 = 0.5 * (2.0 * p4 * d4
                 - 1.5 * mu2 * (d4 * p2 + p4 * d2)
                 + 3.0 * mu4 * p2 * d2)
    K_L = d2 * (p4 - 0.5 * mu2) + p2 * d4
    D = ((2.0 / 3.0) * (d3 * p4 + p3 * d4)
         - (5.0 / 3.0) * mu2 * (d3 * p2 + p3 * d2)
         + mu2 * (d1 * p4 + p1 * d4)
         + 1.5 * mu4 * (d1 * p2 + p1 * d2))
    return K_T, K_1, K_L, D


class CollectivePitchComp(om.ExplicitComponent):
    """Closed-form theta_0 from C_T/sigma, or the reverse."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', values=('collective', 'thrust'),
                             default='collective')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('lambda_p', shape=(nn,), desc="inflow ratio lambda'")
        self.add_input('B', shape=(nn,), val=1.0, desc='tip loss factor')
        self.add_input('theta_1', val=-0.17453, units='rad', desc='blade twist')
        self.add_input('a', val=6.0, units='1/rad', desc='lift curve slope')
        self.add_input('x_0', val=0.0, desc='root cutout r/R')

        self.solve_for = ('theta_0' if self.options['mode'] == 'collective'
                          else 'CT_sigma')
        self.given = 'CT_sigma' if self.solve_for == 'theta_0' else 'theta_0'

        units = {'theta_0': 'rad', 'CT_sigma': None}
        self.add_input(self.given, shape=(nn,), units=units[self.given])
        self.add_output(self.solve_for, shape=(nn,), units=units[self.solve_for])

        for name in ('mu', 'lambda_p', 'B', self.given):
            self.declare_partials(self.solve_for, name, rows=ar, cols=ar)
        for name in ('theta_1', 'a', 'x_0'):
            self.declare_partials(self.solve_for, name, rows=ar, cols=zeros)

    def compute(self, inputs, outputs):
        mu, B = inputs['mu'], inputs['B']
        theta_1, a, x_0 = inputs['theta_1'][0], inputs['a'][0], inputs['x_0'][0]
        K_T, K_1, K_L, D = _coefficients(mu, B, x_0, a)
        rest = K_1 * theta_1 + K_L * inputs['lambda_p']

        if self.solve_for == 'theta_0':
            outputs['theta_0'] = (K_T * inputs['CT_sigma'] - rest) / D
        else:
            outputs['CT_sigma'] = (D * inputs['theta_0'] + rest) / K_T

    def compute_partials(self, inputs, partials):
        out, given = self.solve_for, self.given
        mu, B, lam = inputs['mu'], inputs['B'], inputs['lambda_p']
        theta_1, a, x_0 = inputs['theta_1'][0], inputs['a'][0], inputs['x_0'][0]

        K_T, K_1, K_L, D = _coefficients(mu, B, x_0, a)
        num, den = (K_T, D) if out == 'theta_0' else (D, K_T)
        sign = -1.0 if out == 'theta_0' else 1.0
        value = (num * inputs[given] + sign * (K_1 * theta_1 + K_L * lam)) / den

        partials[out, given] = num / den
        partials[out, 'theta_1'] = sign * K_1 / den
        partials[out, 'lambda_p'] = sign * K_L / den

        for name in ('mu', 'B', 'x_0'):
            dK_T, dK_1, dK_L, dD = _d_coefficients(mu, B, x_0, a, name)
            d_num, d_den = (dK_T, dD) if out == 'theta_0' else (dD, dK_T)
            partials[out, name] = (
                d_num * inputs[given] + sign * (dK_1 * theta_1 + dK_L * lam)
                - value * d_den) / den

        # `a` enters only through K_T, as 1/a
        d_num, d_den = (-K_T / a, 0.0) if out == 'theta_0' else (0.0, -K_T / a)
        partials[out, 'a'] = (d_num * inputs[given] - value * d_den) / den
