"""
TorqueCoefComp -- closed-form rotor torque coefficient.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 174-175 and p. 200 (tip loss and root cutout).

    C_Q/sigma = (c_d/8)(1 + mu^2)                                   profile
                - (a/4) [ lambda' G_A / D_A + mu^2 G_B / D_B ]      inflow

with p_n = B^n - x_0^n and

    D_A = p4 + (3/2) mu^2 p2
    D_B = p4 + (1/2) mu^2 p2
    G_A = [ (2/3) theta_0 p3 + (theta_1/2) p4 ] [ p4 - (mu^2/2) p2 ]
          + lambda' p2 [ p4 + (mu^2/2) p2 ]
    G_B = a_0^2 [ -(4/9) p3^2 + p2 p4 / 2 + p2^2 mu^4/(4 mu^2) ... ]
          + (1/3) a_0 mu (v1/OmegaR) p3 p2 + (1/8) (v1/OmegaR)^2 p2^2

(the a_0^2 bracket is -(4/9) p3^2 + p2 p4/2 + p2^2 mu^2/4).

The equation printed on p. 174 still carries every cyclic and flapping term.
Substituting the expressions for B_1 + a_1s and A_1 - b_1s makes all of them
drop out (p. 175) -- which is why this component needs no cyclic input at all,
only theta_0, theta_1, lambda', a_0 and v1/OmegaR.

Structure of the two parts:

  * Profile, (c_d/8)(1 + mu^2). No tip loss and no root cutout are applied to
    it, deliberately: Prouty keeps the pressure drag acting over the whole
    blade, the same choice he makes again in the numerical method (p. 212).
    Replacing this single term by the three-term polar of p. 205 is what the
    drag_model option will do in the eliminating-the-assumptions pass; it is
    isolated in _profile() for that reason.

  * Inflow. G_A collects the induced and parasite contributions through
    lambda', G_B the coning and induced velocity contributions. G_B is the
    term Prouty calls generally negligible (p. 175): it is 0.6 % of C_Q/sigma
    for the example helicopter in level flight. It is kept because it costs
    nothing and it is not negligible in autorotation, where lambda' changes
    sign and G_A collapses.

Setting B = 1 and x_0 = 0 collapses all four groups to the p. 175 form
exactly -- (theta_0/3)(2 - mu^2) + (theta_1/2)(1 - mu^2/2) + lambda'(1 +
mu^2/2) for G_A, and (a_0^2/2)(1/9 + mu^2/2) + (1/3) mu a_0 v1/OmegaR +
(1/8)(v1/OmegaR)^2 for G_B, both verified in the tests.

Sign: C_Q/sigma is positive in powered flight and negative in autorotation,
where the rotor extracts energy from the airflow.

    cd_bar, a, mu, theta_0, theta_1, lambda_p, a0, vi_OR, B, x_0 --> CQ_sigma
"""

import numpy as np
import openmdao.api as om

from .collective_pitch_comp import _powers


class TorqueCoefComp(om.ExplicitComponent):
    """Closed-form C_Q/sigma."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('cd_bar', shape=(nn,), desc='mean drag coefficient')
        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('theta_0', shape=(nn,), units='rad', desc='collective')
        self.add_input('lambda_p', shape=(nn,), desc="inflow ratio lambda'")
        self.add_input('a0', shape=(nn,), units='rad', desc='coning angle')
        self.add_input('vi_OR', shape=(nn,), desc='v1 / (Omega R)')
        self.add_input('B', shape=(nn,), val=1.0, desc='tip loss factor')
        self.add_input('theta_1', val=-0.17453, units='rad', desc='blade twist')
        self.add_input('a', val=6.0, units='1/rad', desc='lift curve slope')
        self.add_input('x_0', val=0.0, desc='root cutout r/R')

        self.add_output('CQ_sigma', shape=(nn,), desc='C_Q / sigma')

        for name in ('cd_bar', 'mu', 'theta_0', 'lambda_p', 'a0', 'vi_OR', 'B'):
            self.declare_partials('CQ_sigma', name, rows=ar, cols=ar)
        for name in ('theta_1', 'a', 'x_0'):
            self.declare_partials('CQ_sigma', name, rows=ar, cols=zeros)

    @staticmethod
    def _profile(cd_bar, mu):
        """Profile torque, p. 175. Isolated for the future drag_model option."""
        return cd_bar / 8.0 * (1.0 + mu ** 2)

    def _groups(self, inputs):
        """D_A, D_B, G_A, G_B and the powers p_n, p. 200."""
        mu, mu2 = inputs['mu'], inputs['mu'] ** 2
        p, dB, dx = _powers(inputs['B'], inputs['x_0'][0])
        p1, p2, p3, p4 = p
        th0, th1 = inputs['theta_0'], inputs['theta_1'][0]
        lam, a0, vi = inputs['lambda_p'], inputs['a0'], inputs['vi_OR']

        D_A = p4 + 1.5 * mu2 * p2
        D_B = p4 + 0.5 * mu2 * p2
        G_A = ((2.0 / 3.0) * th0 * p3 + 0.5 * th1 * p4) * (p4 - 0.5 * mu2 * p2) \
            + lam * p2 * (p4 + 0.5 * mu2 * p2)
        G_B = (a0 ** 2 * (-(4.0 / 9.0) * p3 ** 2 + 0.5 * p2 * p4
                          + 0.25 * mu2 * p2 ** 2)
               + (1.0 / 3.0) * a0 * mu * vi * p3 * p2
               + 0.125 * vi ** 2 * p2 ** 2)
        return D_A, D_B, G_A, G_B, p, dB, dx

    def _inflow(self, inputs):
        D_A, D_B, G_A, G_B, *_ = self._groups(inputs)
        return (inputs['lambda_p'] * G_A / D_A
                + inputs['mu'] ** 2 * G_B / D_B)

    def compute(self, inputs, outputs):
        outputs['CQ_sigma'] = (self._profile(inputs['cd_bar'], inputs['mu'])
                               - 0.25 * inputs['a'][0] * self._inflow(inputs))

    def compute_partials(self, inputs, partials):
        out = 'CQ_sigma'
        mu, mu2 = inputs['mu'], inputs['mu'] ** 2
        a, th0, th1 = inputs['a'][0], inputs['theta_0'], inputs['theta_1'][0]
        lam, a0, vi = inputs['lambda_p'], inputs['a0'], inputs['vi_OR']
        D_A, D_B, G_A, G_B, p, dB, dx = self._groups(inputs)
        p1, p2, p3, p4 = p
        k = -0.25 * a
        span = p4 - 0.5 * mu2 * p2                      # recurring bracket

        partials[out, 'cd_bar'] = (1.0 + mu2) / 8.0
        partials[out, 'a'] = -0.25 * self._inflow(inputs)

        partials[out, 'theta_0'] = k * lam / D_A * (2.0 / 3.0) * p3 * span
        partials[out, 'theta_1'] = k * lam / D_A * 0.5 * p4 * span
        partials[out, 'lambda_p'] = k * (
            G_A + lam * p2 * (p4 + 0.5 * mu2 * p2)) / D_A
        partials[out, 'a0'] = k * mu2 / D_B * (
            2.0 * a0 * (-(4.0 / 9.0) * p3 ** 2 + 0.5 * p2 * p4
                        + 0.25 * mu2 * p2 ** 2)
            + (1.0 / 3.0) * mu * vi * p3 * p2)
        partials[out, 'vi_OR'] = k * mu2 / D_B * (
            (1.0 / 3.0) * a0 * mu * p3 * p2 + 0.25 * vi * p2 ** 2)

        # ------------------------------------------------------------- mu
        dD_A, dD_B = 3.0 * mu * p2, mu * p2
        dG_A = (((2.0 / 3.0) * th0 * p3 + 0.5 * th1 * p4) * (-mu * p2)
                + lam * p2 * mu * p2)
        dG_B = 0.5 * a0 ** 2 * mu * p2 ** 2 + (1.0 / 3.0) * a0 * vi * p3 * p2
        partials[out, 'mu'] = (
            inputs['cd_bar'] * mu / 4.0
            + k * (lam * (dG_A * D_A - G_A * dD_A) / D_A ** 2
                   + (2.0 * mu * G_B + mu2 * dG_B) / D_B
                   - mu2 * G_B * dD_B / D_B ** 2))

        # ---------------------------------------------------------- B, x_0
        for name, (d1, d2, d3, d4) in (('B', dB), ('x_0', dx)):
            dD_A = d4 + 1.5 * mu2 * d2
            dD_B = d4 + 0.5 * mu2 * d2
            dG_A = (((2.0 / 3.0) * th0 * d3 + 0.5 * th1 * d4) * span
                    + ((2.0 / 3.0) * th0 * p3 + 0.5 * th1 * p4)
                    * (d4 - 0.5 * mu2 * d2)
                    + lam * d2 * (p4 + 0.5 * mu2 * p2)
                    + lam * p2 * (d4 + 0.5 * mu2 * d2))
            dG_B = (a0 ** 2 * (-(8.0 / 9.0) * p3 * d3
                               + 0.5 * (d2 * p4 + p2 * d4)
                               + 0.5 * mu2 * p2 * d2)
                    + (1.0 / 3.0) * a0 * mu * vi * (d3 * p2 + p3 * d2)
                    + 0.25 * vi ** 2 * p2 * d2)
            partials[out, name] = k * (
                lam * (dG_A * D_A - G_A * dD_A) / D_A ** 2
                + mu2 * (dG_B * D_B - G_B * dD_B) / D_B ** 2)
