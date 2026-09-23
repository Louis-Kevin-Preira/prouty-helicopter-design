"""
ConingComp -- steady blade flapping, or coning angle.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 169-171 (derivation) and p. 213 (form used numerically).

The coning angle is obtained by summing the aerodynamic, weight, inertia and
centrifugal moments to zero at the flapping hinge and keeping only the terms
free of psi (p. 170-171).

form = 'exact'  (p. 171, first result)

    a0 = (gamma/6) { theta_0 [3/4 + 3 mu^2/4]
                   + theta_1 [3/5 +   mu^2/2]
                   + lambda }                    - (3/2) g R / (Omega R)^2

    where lambda is the CONTROL PLANE inflow, mu[alpha_TPP - (B_1 + a_1s)]
    - v1/Omega R, not the lambda' of the tip path plane. The two differ by
    mu (B_1 + a_1s), which is 0.026 at mu = 0.3 -- half of lambda itself.
    This component rebuilds it as lambda = lambda' - mu (B_1 + a_1s).

form = 'ct'  (p. 171, "for all practical purposes", and p. 213)

    a0 = (2/3) gamma (C_T/sigma) / a - (3/2) g R / (Omega R)^2

    Prouty gets it by recognising the bracket as the C_T/sigma equation of
    p. 165. The recognition is not exact: the theta_0 coefficient is 3/4 +
    3mu^2/4 in one and 2/3 + mu^2 in the other, an 8 % difference at mu = 0.3.
    It is nonetheless the form Prouty actually uses -- Table 3.2 (p. 193) and
    the numerical method of p. 213 both follow it -- so it is the default here.
    It also avoids a coupling, since it needs no theta_0 and no cyclic.

Two book errors worth knowing:

1. p. 171 prints the gravity term as (3/2) g R^2 / (Omega R)^2, in both the
   exact and the approximate equation. That is dimensionally wrong -- it has
   units of length -- and would swamp the aerodynamic term. The correct
   exponent, R rather than R^2, appears on p. 213 and follows from
   m g R^2 / (2 I_b Omega^2) with I_b = m R^3 / 3. This component uses g R.

2. The gravity term assumes a uniform spanwise mass distribution through
   I_b = m R^3 / 3. For a blade whose mass is concentrated outboard, the
   correct term is W_b g R / (2 I_b Omega^2). The difference is small
   -- the whole term is 4.6 % of a0 for the example helicopter -- but it is an
   assumption, not an identity.

    gamma, CT_sigma, a, R, V_tip                       --> 'ct'    --> a0
    gamma, theta_0, theta_1, lambda_p, B1_a1s, mu, ... --> 'exact' --> a0
"""

import numpy as np
import openmdao.api as om


class ConingComp(om.ExplicitComponent):
    """Coning angle a0, closed form."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('form', values=('ct', 'exact'), default='ct',
                             desc="'ct': p. 171/213 practical form; "
                                  "'exact': p. 171 full bracket")

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)

        self.add_input('gamma', shape=(nn,), desc='Lock number')
        self.add_input('g', val=32.174, units='ft/s**2', desc='gravity')
        self.add_input('R', val=30.0, units='ft', desc='rotor radius')
        self.add_input('V_tip', val=650.0, units='ft/s', desc='tip speed')

        self.add_output('a0', shape=(nn,), units='rad', desc='coning angle')

        self.declare_partials('a0', 'gamma', rows=ar, cols=ar)
        for name in ('g', 'R', 'V_tip'):
            self.declare_partials('a0', name, rows=ar, cols=zeros)

        if self.options['form'] == 'ct':
            self.add_input('CT_sigma', shape=(nn,), desc='C_T / sigma')
            self.add_input('a', val=6.0, units='1/rad',
                           desc='reference lift curve slope')
            self.declare_partials('a0', 'CT_sigma', rows=ar, cols=ar)
            self.declare_partials('a0', 'a', rows=ar, cols=zeros)
        else:
            self.add_input('mu', shape=(nn,), desc='tip speed ratio')
            self.add_input('theta_0', shape=(nn,), units='rad',
                           desc='collective pitch')
            self.add_input('lambda_p', shape=(nn,), desc="inflow ratio lambda'")
            self.add_input('B1_a1s', shape=(nn,), units='rad',
                           desc='B_1 + a_1s, longitudinal control plane tilt')
            self.add_input('theta_1', val=-0.17453, units='rad',
                           desc='linear blade twist')
            for name in ('mu', 'theta_0', 'lambda_p', 'B1_a1s'):
                self.declare_partials('a0', name, rows=ar, cols=ar)
            self.declare_partials('a0', 'theta_1', rows=ar, cols=zeros)

    def _gravity(self, inputs):
        """(3/2) g R / (Omega R)^2, p. 213."""
        return 1.5 * inputs['g'][0] * inputs['R'][0] / inputs['V_tip'][0] ** 2

    def compute(self, inputs, outputs):
        if self.options['form'] == 'ct':
            aero = (2.0 / 3.0) * inputs['gamma'] * inputs['CT_sigma'] \
                / inputs['a'][0]
        else:
            aero = inputs['gamma'] / 6.0 * self._bracket(inputs)

        outputs['a0'] = aero - self._gravity(inputs)

    def _bracket(self, inputs):
        """Braced term of the exact equation, p. 171."""
        mu = inputs['mu']
        lambda_cp = inputs['lambda_p'] - mu * inputs['B1_a1s']
        return (inputs['theta_0'] * (0.75 + 0.75 * mu ** 2)
                + inputs['theta_1'][0] * (0.6 + 0.5 * mu ** 2)
                + lambda_cp)

    def compute_partials(self, inputs, partials):
        g, R, V_tip = inputs['g'][0], inputs['R'][0], inputs['V_tip'][0]
        gamma = inputs['gamma']

        partials['a0', 'g'] = -1.5 * R / V_tip ** 2
        partials['a0', 'R'] = -1.5 * g / V_tip ** 2
        partials['a0', 'V_tip'] = 3.0 * g * R / V_tip ** 3

        if self.options['form'] == 'ct':
            CT_sigma, a = inputs['CT_sigma'], inputs['a'][0]
            partials['a0', 'gamma'] = (2.0 / 3.0) * CT_sigma / a
            partials['a0', 'CT_sigma'] = (2.0 / 3.0) * gamma / a
            partials['a0', 'a'] = -(2.0 / 3.0) * gamma * CT_sigma / a ** 2
        else:
            mu = inputs['mu']
            partials['a0', 'gamma'] = self._bracket(inputs) / 6.0
            partials['a0', 'theta_0'] = gamma / 6.0 * (0.75 + 0.75 * mu ** 2)
            partials['a0', 'theta_1'] = gamma / 6.0 * (0.6 + 0.5 * mu ** 2)
            partials['a0', 'lambda_p'] = gamma / 6.0
            partials['a0', 'B1_a1s'] = -gamma / 6.0 * mu
            partials['a0', 'mu'] = gamma / 6.0 * (
                1.5 * mu * inputs['theta_0'] + mu * inputs['theta_1'][0]
                - inputs['B1_a1s'])
