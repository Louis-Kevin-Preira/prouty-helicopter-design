"""
LongFlappingComp -- longitudinal flapping and cyclic.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 168 (tip path plane form), p. 169 (shaft form), p. 200 (tip loss
and root cutout).

form = 'tpp'  (p. 168, generalised p. 200)

    B_1 + a_1s = mu [ (8/3) p3 theta_0 + 2 p4 theta_1 + 2 p2 lambda' ]
                 / [ p4 + (3/2) p2 mu^2 ]          with p_n = B^n - x_0^n

    This is the longitudinal tilt of the CONTROL plane relative to the tip
    path plane. Only the sum is determined by the aerodynamics: splitting it
    into cyclic B_1 and flapping a_1s takes a pitching moment balance, which
    belongs to Chapter 8. Table 3.2 sets a_1s = 0, so B_1 there IS the sum --
    valid for performance, not for stability and control.

    At B = 1, x_0 = 0 this collapses to mu (8/3 theta_0 + 2 theta_1 +
    2 lambda') / (1 + 1.5 mu^2), the p. 168 form. The denominator is also
    (a/4) K_T of CollectivePitchComp: the same integral appears in both.

form = 'shaft'  (p. 169)

    a_1s = mu [ (8/3) theta_0 + 2 theta_1 + 2 (mu alpha_s - v1/Omega R) ]
           / (1 - mu^2/2)
           - [ (1 + 1.5 mu^2) / (1 - mu^2/2) ] B_1

    Here the cyclic B_1 is prescribed and the flapping a_1s is the unknown.
    That is the situation of a rotor in a wind tunnel with fixed controls, and
    of the tail rotor of p. 187-191, whose flapping responds to shaft angle
    rather than being trimmed out. The reference is the shaft, alpha_s, not
    the tip path plane.

    The two forms are one relation seen from two planes, related by
    alpha_TPP = alpha_s + a_1s; substituting it into 'tpp' and solving for
    a_1s reproduces 'shaft' exactly, denominator 1 - mu^2/2 included. That
    identity is checked in the tests.

Prouty gives the B, x_0 generalisation for 'tpp' only (p. 200), so 'shaft'
carries no tip loss.

    mu, theta_0, theta_1, lambda_p, B, x_0        --> 'tpp'   --> B1_a1s
    mu, theta_0, theta_1, alpha_s, vi_OR, B_1     --> 'shaft' --> a1s
"""

import numpy as np
import openmdao.api as om

from .collective_pitch_comp import _powers


class LongFlappingComp(om.ExplicitComponent):
    """Longitudinal flapping, referred to the tip path plane or to the shaft."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('form', values=('tpp', 'shaft'), default='tpp')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('theta_0', shape=(nn,), units='rad', desc='collective')
        self.add_input('theta_1', val=-0.17453, units='rad', desc='blade twist')

        if self.options['form'] == 'tpp':
            self.add_input('lambda_p', shape=(nn,), desc="inflow ratio lambda'")
            self.add_input('B', shape=(nn,), val=1.0, desc='tip loss factor')
            self.add_input('x_0', val=0.0, desc='root cutout r/R')
            self.add_output('B1_a1s', shape=(nn,), units='rad',
                            desc='B_1 + a_1s')

            for name in ('mu', 'theta_0', 'lambda_p', 'B'):
                self.declare_partials('B1_a1s', name, rows=ar, cols=ar)
            for name in ('theta_1', 'x_0'):
                self.declare_partials('B1_a1s', name, rows=ar, cols=zeros)
        else:
            self.add_input('alpha_s', shape=(nn,), units='rad',
                           desc='shaft angle of attack')
            self.add_input('vi_OR', shape=(nn,), desc='v1 / (Omega R)')
            self.add_input('B_1', shape=(nn,), units='rad',
                           desc='longitudinal cyclic pitch')
            self.add_output('a1s', shape=(nn,), units='rad',
                            desc='longitudinal flapping')

            for name in ('mu', 'theta_0', 'alpha_s', 'vi_OR', 'B_1'):
                self.declare_partials('a1s', name, rows=ar, cols=ar)
            self.declare_partials('a1s', 'theta_1', rows=ar, cols=zeros)

    # ------------------------------------------------------------------ tpp
    def _tpp(self, inputs):
        mu, B = inputs['mu'], inputs['B']
        (p1, p2, p3, p4), _, _ = _powers(B, inputs['x_0'][0])
        S = ((8.0 / 3.0) * p3 * inputs['theta_0']
             + 2.0 * p4 * inputs['theta_1'][0]
             + 2.0 * p2 * inputs['lambda_p'])
        return S, p2, p4, p4 + 1.5 * mu ** 2 * p2

    # ---------------------------------------------------------------- shaft
    def _shaft(self, inputs):
        mu = inputs['mu']
        S = ((8.0 / 3.0) * inputs['theta_0'] + 2.0 * inputs['theta_1'][0]
             + 2.0 * (mu * inputs['alpha_s'] - inputs['vi_OR']))
        return S, 1.0 - 0.5 * mu ** 2

    def compute(self, inputs, outputs):
        mu = inputs['mu']
        if self.options['form'] == 'tpp':
            S, _, _, D = self._tpp(inputs)
            outputs['B1_a1s'] = mu * S / D
        else:
            S, E = self._shaft(inputs)
            outputs['a1s'] = (mu * S - (1.0 + 1.5 * mu ** 2) * inputs['B_1']) / E

    def compute_partials(self, inputs, partials):
        mu = inputs['mu']

        if self.options['form'] == 'tpp':
            out = 'B1_a1s'
            theta_1, x_0 = inputs['theta_1'][0], inputs['x_0'][0]
            S, p2, p4, D = self._tpp(inputs)
            value = mu * S / D
            (_, _, p3_, _), dB, dx = _powers(inputs['B'], x_0)

            partials[out, 'theta_0'] = mu * (8.0 / 3.0) * p3_ / D
            partials[out, 'theta_1'] = 2.0 * mu * p4 / D
            partials[out, 'lambda_p'] = 2.0 * mu * p2 / D
            partials[out, 'mu'] = (S - value * 3.0 * mu * p2) / D

            for name, (d1, d2, d3, d4) in (('B', dB), ('x_0', dx)):
                dS = ((8.0 / 3.0) * d3 * inputs['theta_0'] + 2.0 * d4 * theta_1
                      + 2.0 * d2 * inputs['lambda_p'])
                dD = d4 + 1.5 * mu ** 2 * d2
                partials[out, name] = (mu * dS - value * dD) / D
        else:
            out = 'a1s'
            S, E = self._shaft(inputs)
            value = (mu * S - (1.0 + 1.5 * mu ** 2) * inputs['B_1']) / E

            partials[out, 'theta_0'] = mu * (8.0 / 3.0) / E
            partials[out, 'theta_1'] = 2.0 * mu / E
            partials[out, 'alpha_s'] = 2.0 * mu ** 2 / E
            partials[out, 'vi_OR'] = -2.0 * mu / E
            partials[out, 'B_1'] = -(1.0 + 1.5 * mu ** 2) / E
            partials[out, 'mu'] = (
                (S + 2.0 * mu * inputs['alpha_s']
                 - 3.0 * mu * inputs['B_1']) + value * mu) / E
