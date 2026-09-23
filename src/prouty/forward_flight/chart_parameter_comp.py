"""
ChartParameterComp -- the ordinate of the isolated rotor charts.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 229-231, charts p. 254-271; the same group appears as step e of
Table 3.5 case 4, p. 235.

    X = f/A_b + (sigma / mu^4) (C_T/sigma)^2

The charts plot X against C_T/sigma, with curves of constant collective. The
printed coefficient changes from chart to chart -- 10,000 at mu = 0.10, 123 at
mu = 0.30 -- because it is 1/mu^4 each time.

X is lambda' in disguise. For an isolated rotor the tip path plane carries only
the parasite drag, alpha_TPP = -q f / T, so with q = (rho/2)(mu Omega R)^2,
T = (C_T/sigma) rho A_b (Omega R)^2 and the momentum inflow
v_1/Omega R = sigma (C_T/sigma) / 2 mu,

    lambda' = mu alpha_TPP - v_1/Omega R
            = -(1/2 mu) [ mu^4 f/(A_b C_T/sigma) + sigma (C_T/sigma) ]

and multiplying by -2 (C_T/sigma) / mu^3 gives exactly X. The identity holds to
machine precision, and it is what makes the charts usable at all: G2 takes
lambda' as an input and returns C_T/sigma, so every (lambda', theta_0) solution
lands on a chart point without anything else being computed.

    X = -2 lambda' (C_T/sigma) / mu^3

Two directions, and both are needed. Generating a chart means running G2 on a
grid of lambda' and theta_0 and converting each result to X, which is
mode='from_inflow'. Using a chart, or driving G2 to a prescribed flight
condition, means starting from the airframe -- f/A_b and sigma are known, C_T
follows from the weight -- and getting the lambda' that puts the rotor there,
which is mode='to_inflow'. Table 3.5 case 4 works in the second direction.

Singular at mu = 0 and at C_T/sigma = 0. The first is the hovering limit, where
the whole forward flight formulation goes away. The second is not: a rotor at
zero thrust has a perfectly good lambda', but the chart ordinate diverges
because the parasite drag term carries 1/(C_T/sigma). The charts start at
C_T/sigma = 0 with all the theta_0 curves converging on X = 0, which is the
other branch of the same limit -- X -> 0 as C_T/sigma -> 0 at fixed lambda'.
Which branch you are on depends on which variable is held fixed, so guard
whichever one you are dividing by rather than assuming the other is safe.

    lambda_p, CT_sigma, mu           --> 'from_inflow' --> X_chart
    f_Ab, sigma, CT_sigma, mu        --> 'to_inflow'   --> X_chart, lambda_p
"""

import numpy as np
import openmdao.api as om


class ChartParameterComp(om.ExplicitComponent):
    """Isolated rotor chart ordinate and its link to the inflow, p. 229."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('mode', values=('from_inflow', 'to_inflow'),
                             default='from_inflow')
        self.options.declare('epsilon', types=float, default=1.0e-8,
                             desc='guard on C_T/sigma in the to_inflow branch')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        forward = self.options['mode'] == 'from_inflow'

        self.add_input('CT_sigma', shape=(nn,), desc='C_T/sigma')
        self.add_input('mu', shape=(nn,), desc='tip speed ratio')

        self.add_output('X_chart', shape=(nn,),
                        desc='f/A_b + (sigma/mu^4)(C_T/sigma)^2, p. 254')

        if forward:
            self.add_input('lambda_p', shape=(nn,), desc="inflow ratio")
            self.declare_partials('X_chart', ['lambda_p', 'CT_sigma', 'mu'],
                                  rows=ar, cols=ar)
        else:
            self.add_input('f_Ab', val=19.4 / 240.0,
                           desc='equivalent flat plate area over blade area')
            self.add_input('sigma', val=0.084883, desc='solidity')
            self.add_output('lambda_p', shape=(nn,), desc="inflow ratio")

            for out in ('X_chart', 'lambda_p'):
                self.declare_partials(out, ['CT_sigma', 'mu'], rows=ar, cols=ar)
                self.declare_partials(out, ['f_Ab', 'sigma'], rows=ar,
                                      cols=zeros)

    def _guard(self, value):
        eps = self.options['epsilon']
        return np.where(np.real(value) >= 0.0, value + eps, value - eps)

    def compute(self, inputs, outputs):
        CT, mu = inputs['CT_sigma'], inputs['mu']

        if self.options['mode'] == 'from_inflow':
            outputs['X_chart'] = -2.0 * inputs['lambda_p'] * CT / mu ** 3
        else:
            safe = self._guard(CT)
            X = inputs['f_Ab'][0] + inputs['sigma'][0] * CT ** 2 / mu ** 4
            outputs['X_chart'] = X
            outputs['lambda_p'] = -0.5 * X * mu ** 3 / safe

    def compute_partials(self, inputs, partials):
        CT, mu = inputs['CT_sigma'], inputs['mu']

        if self.options['mode'] == 'from_inflow':
            lam = inputs['lambda_p']
            partials['X_chart', 'lambda_p'] = -2.0 * CT / mu ** 3
            partials['X_chart', 'CT_sigma'] = -2.0 * lam / mu ** 3
            partials['X_chart', 'mu'] = 6.0 * lam * CT / mu ** 4
            return

        f_Ab, sigma = inputs['f_Ab'][0], inputs['sigma'][0]
        safe = self._guard(CT)
        X = f_Ab + sigma * CT ** 2 / mu ** 4

        dX = {'CT_sigma': 2.0 * sigma * CT / mu ** 4,
              'mu': -4.0 * sigma * CT ** 2 / mu ** 5,
              'f_Ab': np.ones_like(CT),
              'sigma': CT ** 2 / mu ** 4}
        for name, value in dX.items():
            partials['X_chart', name] = value

        # lambda' = -X mu^3 / (2 C_T/sigma)
        scale = -0.5 * mu ** 3 / safe
        partials['lambda_p', 'CT_sigma'] = (scale * dX['CT_sigma']
                                            + 0.5 * X * mu ** 3 / safe ** 2)
        partials['lambda_p', 'mu'] = (scale * dX['mu']
                                      - 1.5 * X * mu ** 2 / safe)
        partials['lambda_p', 'f_Ab'] = scale * dX['f_Ab']
        partials['lambda_p', 'sigma'] = scale * dX['sigma']
