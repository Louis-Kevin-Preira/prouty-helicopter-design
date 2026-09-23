"""
HForceCoefComp -- closed-form rotor H-force coefficient.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 175-176 and p. 201 (tip loss, root cutout, and the identity
form).

The H-force is the in-plane force perpendicular to the shaft. Prouty warns
(p. 176) that it "can vary drastically with relatively small changes in flight
conditions", because the advancing and retreating sides contribute with
opposite signs and the total is their difference. That sensitivity is the
reason both forms below are provided rather than one.

form = 'direct'  (p. 176, generalised p. 201)

    C_H/sigma = (c_d/4) mu
                - (a/4) [ lambda' mu H_A / D_A - mu H_B / D_B ]
                + a_1s (C_T/sigma)

    with p_n = B^n - x_0^n, D_A = p4 + (3/2) mu^2 p2, D_B = p4 + (1/2) mu^2 p2,

    H_A = theta_0 [ p1 p4 + (3/2) mu^2 p1 p2 - (4/3) p2 p3 ]
        + theta_1 [ -(1/2) p2 p4 + (3/4) mu^2 p2^2 ]
        - lambda' p2^2
    H_B = a_0^2 [ -(4/9) p3^2 + (1/2) p2 p4 + (1/4) mu^2 p2^2 ]
        + (1/3) a_0 mu (v1/OmegaR) p3 p2 + (1/8) (v1/OmegaR)^2 p2 p4

form = 'identity'  (p. 201, the "or" equation; same as p. 197)

    C_H/sigma = (1/mu) [ (c_d/8)(1 + 3 mu^2)
                         - (lambda' - mu a_1s)(C_T/sigma) - C_Q/sigma ]

    This is the p. 177 power identity solved for C_H/sigma. It is what Prouty
    uses to get H_M in the autorotation trim (p. 197). Being algebraically
    tied to whatever C_Q/sigma was actually computed, it can never disagree
    with the torque; the price is a dependency on C_Q/sigma and a singularity
    at mu = 0.

Running both and comparing is the power identity check of p. 177,
-C_Q/sigma_inflow = lambda' C_T/sigma + mu (C_H/sigma_inflow - a_1s C_T/sigma).
No separate component is needed for it -- it is a test, not a computation.

One transcription trap worth recording: the last term of H_B carries p2 p4,
whereas the same-looking term in the torque equation carries p2^2 (p. 200 vs
p. 201). Both collapse to 1 at B = 1, x_0 = 0, so reduction alone cannot tell
them apart -- they were checked against the page directly.

    ... --> HForceCoefComp --> CH_sigma (nn,)
"""

import numpy as np
import openmdao.api as om

from .collective_pitch_comp import _powers


class HForceCoefComp(om.ExplicitComponent):
    """Closed-form C_H/sigma."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('form', values=('direct', 'identity'),
                             default='direct')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('cd_bar', shape=(nn,), desc='mean drag coefficient')
        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('lambda_p', shape=(nn,), desc="inflow ratio lambda'")
        self.add_input('CT_sigma', shape=(nn,), desc='C_T / sigma')
        self.add_input('a1s', shape=(nn,), val=0.0, units='rad',
                       desc='longitudinal flapping')

        self.add_output('CH_sigma', shape=(nn,), desc='C_H / sigma')

        common = ['cd_bar', 'mu', 'lambda_p', 'CT_sigma', 'a1s']

        if self.options['form'] == 'direct':
            self.add_input('theta_0', shape=(nn,), units='rad',
                           desc='collective')
            self.add_input('a0', shape=(nn,), units='rad', desc='coning angle')
            self.add_input('vi_OR', shape=(nn,), desc='v1 / (Omega R)')
            self.add_input('B', shape=(nn,), val=1.0, desc='tip loss factor')
            self.add_input('theta_1', val=-0.17453, units='rad', desc='twist')
            self.add_input('a', val=6.0, units='1/rad', desc='lift curve slope')
            self.add_input('x_0', val=0.0, desc='root cutout r/R')

            common += ['theta_0', 'a0', 'vi_OR', 'B']
            for name in ('theta_1', 'a', 'x_0'):
                self.declare_partials('CH_sigma', name, rows=ar, cols=zeros)
        else:
            self.add_input('CQ_sigma', shape=(nn,), desc='C_Q / sigma')
            common += ['CQ_sigma']

        for name in common:
            self.declare_partials('CH_sigma', name, rows=ar, cols=ar)

    # -------------------------------------------------------------- direct
    def _groups(self, inputs):
        mu, mu2 = inputs['mu'], inputs['mu'] ** 2
        p, dB, dx = _powers(inputs['B'], inputs['x_0'][0])
        p1, p2, p3, p4 = p
        a0, vi = inputs['a0'], inputs['vi_OR']

        K_a = p1 * p4 + 1.5 * mu2 * p1 * p2 - (4.0 / 3.0) * p2 * p3
        K_b = -0.5 * p2 * p4 + 0.75 * mu2 * p2 ** 2
        H_A = (inputs['theta_0'] * K_a + inputs['theta_1'][0] * K_b
               - inputs['lambda_p'] * p2 ** 2)
        H_B = (a0 ** 2 * (-(4.0 / 9.0) * p3 ** 2 + 0.5 * p2 * p4
                          + 0.25 * mu2 * p2 ** 2)
               + (1.0 / 3.0) * a0 * mu * vi * p3 * p2
               + 0.125 * vi ** 2 * p2 * p4)
        return (K_a, K_b, H_A, H_B, p4 + 1.5 * mu2 * p2,
                p4 + 0.5 * mu2 * p2, p, dB, dx)

    def _inflow(self, inputs):
        _, _, H_A, H_B, D_A, D_B, *_ = self._groups(inputs)
        mu = inputs['mu']
        return inputs['lambda_p'] * mu * H_A / D_A - mu * H_B / D_B

    def compute(self, inputs, outputs):
        mu, cd = inputs['mu'], inputs['cd_bar']

        if self.options['form'] == 'direct':
            outputs['CH_sigma'] = (0.25 * cd * mu
                                   - 0.25 * inputs['a'][0] * self._inflow(inputs)
                                   + inputs['a1s'] * inputs['CT_sigma'])
        else:
            outputs['CH_sigma'] = (
                cd / 8.0 * (1.0 + 3.0 * mu ** 2)
                - (inputs['lambda_p'] - mu * inputs['a1s']) * inputs['CT_sigma']
                - inputs['CQ_sigma']) / mu

    def compute_partials(self, inputs, partials):
        out = 'CH_sigma'
        mu, mu2 = inputs['mu'], inputs['mu'] ** 2
        cd, lam = inputs['cd_bar'], inputs['lambda_p']
        CT, a1s = inputs['CT_sigma'], inputs['a1s']

        if self.options['form'] == 'identity':
            num = (cd / 8.0 * (1.0 + 3.0 * mu2) - (lam - mu * a1s) * CT
                   - inputs['CQ_sigma'])
            value = num / mu

            partials[out, 'cd_bar'] = (1.0 + 3.0 * mu2) / (8.0 * mu)
            partials[out, 'lambda_p'] = -CT / mu
            partials[out, 'CT_sigma'] = -(lam - mu * a1s) / mu
            partials[out, 'a1s'] = CT
            partials[out, 'CQ_sigma'] = -1.0 / mu
            partials[out, 'mu'] = (0.75 * cd * mu + a1s * CT - value) / mu
            return

        a, th0, th1 = inputs['a'][0], inputs['theta_0'], inputs['theta_1'][0]
        a0, vi = inputs['a0'], inputs['vi_OR']
        K_a, K_b, H_A, H_B, D_A, D_B, p, dB, dx = self._groups(inputs)
        p1, p2, p3, p4 = p
        k = -0.25 * a
        coning = -(4.0 / 9.0) * p3 ** 2 + 0.5 * p2 * p4 + 0.25 * mu2 * p2 ** 2

        partials[out, 'cd_bar'] = 0.25 * mu
        partials[out, 'a'] = -0.25 * self._inflow(inputs)
        partials[out, 'CT_sigma'] = a1s
        partials[out, 'a1s'] = CT

        partials[out, 'theta_0'] = k * lam * mu * K_a / D_A
        partials[out, 'theta_1'] = k * lam * mu * K_b / D_A
        partials[out, 'lambda_p'] = k * mu * (H_A - lam * p2 ** 2) / D_A
        partials[out, 'a0'] = -k * mu / D_B * (
            2.0 * a0 * coning + (1.0 / 3.0) * mu * vi * p3 * p2)
        partials[out, 'vi_OR'] = -k * mu / D_B * (
            (1.0 / 3.0) * a0 * mu * p3 * p2 + 0.25 * vi * p2 * p4)

        # ------------------------------------------------------------- mu
        dK_a, dK_b = 3.0 * mu * p1 * p2, 1.5 * mu * p2 ** 2
        dH_A = th0 * dK_a + th1 * dK_b
        dH_B = 0.5 * a0 ** 2 * mu * p2 ** 2 + (1.0 / 3.0) * a0 * vi * p3 * p2
        dD_A, dD_B = 3.0 * mu * p2, mu * p2
        partials[out, 'mu'] = 0.25 * cd + k * (
            lam * ((H_A + mu * dH_A) * D_A - mu * H_A * dD_A) / D_A ** 2
            - ((H_B + mu * dH_B) * D_B - mu * H_B * dD_B) / D_B ** 2)

        # ---------------------------------------------------------- B, x_0
        for name, (d1, d2, d3, d4) in (('B', dB), ('x_0', dx)):
            dK_a = (d1 * p4 + p1 * d4 + 1.5 * mu2 * (d1 * p2 + p1 * d2)
                    - (4.0 / 3.0) * (d2 * p3 + p2 * d3))
            dK_b = -0.5 * (d2 * p4 + p2 * d4) + 1.5 * mu2 * p2 * d2
            dH_A = th0 * dK_a + th1 * dK_b - lam * 2.0 * p2 * d2
            dH_B = (a0 ** 2 * (-(8.0 / 9.0) * p3 * d3
                               + 0.5 * (d2 * p4 + p2 * d4)
                               + 0.5 * mu2 * p2 * d2)
                    + (1.0 / 3.0) * a0 * mu * vi * (d3 * p2 + p3 * d2)
                    + 0.125 * vi ** 2 * (d2 * p4 + p2 * d4))
            dD_A = d4 + 1.5 * mu2 * d2
            dD_B = d4 + 0.5 * mu2 * d2
            partials[out, name] = k * mu * (
                lam * (dH_A * D_A - H_A * dD_A) / D_A ** 2
                - (dH_B * D_B - H_B * dD_B) / D_B ** 2)
