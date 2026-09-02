"""
RotorDragAreaComp -- the rotor's own equivalent flat plate area.

Reference: Prouty, "Helicopter Performance, Stability and Control",
Chapter 3, p. 244 (Table 3.5, case 7, step s) and p. 246 (case 8, step s).

    f_M = -A_b [ X - (sigma/mu^4)(C_T/sigma)^2 ]

where X is the chart ordinate, f/A_b + (sigma/mu^4)(C_T/sigma)^2, evaluated at
the operating point. Subtracting the second term leaves f/A_b, the equivalent
flat plate area of everything the rotor is pushing against, and the minus sign
turns it into the rotor's own contribution.

Sign, and what f_M actually is. Prouty's note on p. 244 is the key to reading
it: "for level flight, f_M = -(f + H_T/q)". In level flight the rotor produces
a net propulsive force that exactly overcomes the fuselage and tail rotor drag,
so its equivalent flat plate area is NEGATIVE. When the rotor is instead being
dragged -- a dive at low collective, or a compound helicopter whose propeller
does the pushing -- f_M turns positive and something else has to supply
q(f + f_M).

That is why the same number serves two cases. p. 244 step u puts gravity in
that role, G.W. sin(gamma_D) = q(f + f_M) + H_T, and p. 246 step u puts a
propeller there, T_aux = q(f + f_M) + H_T. Prouty flags the similarity himself.

Equivalent forms. With X = -2 lambda' (C_T/sigma)/mu^3, from
ChartParameterComp,

    f_M = 2 A_b lambda' (C_T/sigma)/mu^3 + sigma A_b (C_T/sigma)^2/mu^4

which is what source='inflow' computes. The two agree exactly; use whichever
the caller has. G2 returns lambda', so the inflow form avoids a round trip.

Worked check, p. 244: X = -0.10, C_T/sigma = 0.083, mu = 0.3, sigma = 0.085,
A_b = 240 ft^2 gives f_M = -240[-0.10 - 123(0.085)(0.083^2)] = 41.3 ft^2. The
rotor in that dive contributes 41 ft^2 of drag, twice the fuselage's 19.5.

Singular at mu = 0, like everything else on the chart branch: the 1/mu^4 term
is the parasite drag written in rotor coefficients, and it has no hovering
limit.

    X_chart, CT_sigma, mu, sigma, A_b   --> 'chart'  --> f_M
    lambda_p, CT_sigma, mu, sigma, A_b  --> 'inflow' --> f_M, X_chart
"""

import numpy as np
import openmdao.api as om


class RotorDragAreaComp(om.ExplicitComponent):
    """Equivalent flat plate area of the rotor itself, p. 244."""

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('source', values=('inflow', 'chart'),
                             default='inflow',
                             desc="'inflow' takes lambda', 'chart' takes the "
                                  'chart ordinate directly')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        from_inflow = self.options['source'] == 'inflow'

        self.add_input('CT_sigma', shape=(nn,))
        self.add_input('mu', shape=(nn,))
        self.add_input('sigma', val=0.084883)
        self.add_input('A_b', val=240.0, units='ft**2', desc='blade area')

        self.add_output('f_M', shape=(nn,), units='ft**2',
                        desc="rotor's equivalent flat plate area, negative "
                             'when the rotor is propelling')

        moving = ['CT_sigma', 'mu']
        if from_inflow:
            self.add_input('lambda_p', shape=(nn,))
            self.add_output('X_chart', shape=(nn,))
            moving.append('lambda_p')
            outputs = ['f_M', 'X_chart']
        else:
            self.add_input('X_chart', shape=(nn,))
            moving.append('X_chart')
            outputs = ['f_M']

        for name in moving:
            self.declare_partials(outputs, name, rows=ar, cols=ar)
        self.declare_partials('f_M', ['sigma', 'A_b'], rows=ar, cols=zeros)
        if from_inflow:
            self.declare_partials('X_chart', 'sigma', dependent=False)

    def _parasite(self, inputs):
        """(sigma/mu^4)(C_T/sigma)^2, the rotor-coefficient parasite term."""
        return (inputs['sigma'][0] * inputs['CT_sigma'] ** 2
                / inputs['mu'] ** 4)

    def compute(self, inputs, outputs):
        CT, mu = inputs['CT_sigma'], inputs['mu']
        A_b = inputs['A_b'][0]

        if self.options['source'] == 'inflow':
            X = -2.0 * inputs['lambda_p'] * CT / mu ** 3
            outputs['X_chart'] = X
        else:
            X = inputs['X_chart']

        outputs['f_M'] = -A_b * (X - self._parasite(inputs))

    def compute_partials(self, inputs, partials):
        CT, mu = inputs['CT_sigma'], inputs['mu']
        sigma, A_b = inputs['sigma'][0], inputs['A_b'][0]
        parasite = self._parasite(inputs)

        d_parasite = {'CT_sigma': 2.0 * sigma * CT / mu ** 4,
                      'mu': -4.0 * sigma * CT ** 2 / mu ** 5,
                      'sigma': CT ** 2 / mu ** 4}

        if self.options['source'] == 'inflow':
            lam = inputs['lambda_p']
            X = -2.0 * lam * CT / mu ** 3
            dX = {'lambda_p': -2.0 * CT / mu ** 3,
                  'CT_sigma': -2.0 * lam / mu ** 3,
                  'mu': 6.0 * lam * CT / mu ** 4}
            for name, value in dX.items():
                partials['X_chart', name] = value
        else:
            X = inputs['X_chart']
            dX = {'X_chart': np.ones_like(CT),
                  'CT_sigma': np.zeros_like(CT),
                  'mu': np.zeros_like(CT)}

        for name, value in dX.items():
            partials['f_M', name] = -A_b * (value - d_parasite.get(
                name, np.zeros_like(CT)))
        partials['f_M', 'sigma'] = A_b * d_parasite['sigma']
        partials['f_M', 'A_b'] = -(X - parasite)
