"""
InflowComp -- inflow ratio referred to the tip path plane.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 166-167:

    lambda' = mu alpha_TPP - v1 / Omega R

Two inflow ratios live side by side in this chapter and they are not the same
quantity (p. 165-166):

    lambda  = mu [alpha_TPP - (B_1 + a_1s)] - v1/Omega R    control plane
    lambda' = mu  alpha_TPP                 - v1/Omega R    tip path plane

lambda is the flow perpendicular to the swashplate, the classical variable of
references 3.16 and 3.17. lambda' drops the cyclic term and is the one Prouty
carries through the rest of the chapter: it is the third argument of the
closed-form C_T/sigma, C_Q/sigma and C_H/sigma equations (p. 167, 175, 176),
the ordinate of the middle plot of every second chart page (p. 255), and the
input to the numerical U_P of p. 213. Mixing the two silently costs
mu (B_1 + a_1s), about 0.02 at mu = 0.3 -- comparable to lambda' itself.

Prouty also gives lambda' in fully expanded form on p. 167:

    lambda' = -[ (f/A_b) (mu^3/2) / (C_T/sigma) + (C_T/sigma) sigma / (2 mu) ]

That expression is not coded separately: it is exactly what this component
returns when fed by TppAngleComp(form='parasite') and
InducedVelocityComp(form='high_speed'), which is verified in the tests. Keeping
one general form lets the same component serve the trim loop of p. 192, where
alpha_TPP comes from the force balance and no closed expression exists.

Sign: lambda' is negative in level flight and climb (flow down through the
disc) and turns positive in autorotation, where the rotor is climbing through
its own wake.

    mu, alpha_TPP, vi_OR --> InflowComp --> lambda_p (nn,)
"""

import numpy as np
import openmdao.api as om


class InflowComp(om.ExplicitComponent):
    """Inflow ratio perpendicular to the tip path plane."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('mu', shape=(nn,), desc='tip speed ratio')
        self.add_input('alpha_TPP', shape=(nn,), units='rad',
                       desc='tip path plane angle of attack')
        self.add_input('vi_OR', shape=(nn,), desc='v1 / (Omega R)')

        self.add_output('lambda_p', shape=(nn,), desc="inflow ratio lambda'")

        ar = np.arange(nn)
        for name in ('mu', 'alpha_TPP', 'vi_OR'):
            self.declare_partials('lambda_p', name, rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['lambda_p'] = inputs['mu'] * inputs['alpha_TPP'] - inputs['vi_OR']

    def compute_partials(self, inputs, partials):
        partials['lambda_p', 'mu'] = inputs['alpha_TPP']
        partials['lambda_p', 'alpha_TPP'] = inputs['mu']
        partials['lambda_p', 'vi_OR'] = -np.ones_like(inputs['vi_OR'])
