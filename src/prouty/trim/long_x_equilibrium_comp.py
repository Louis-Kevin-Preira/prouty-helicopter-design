"""X-equilibrium equation in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", Table 8.4 p. 518. Hover form on p. 516.
"""

import numpy as np
import openmdao.api as om


class LongXEquilibriumComp(om.ExplicitComponent):
    """Sum of the X forces, Table 8.4 p. 518::

        res_X = X_M + X_T + X_H + X_V + X_F - G.W. sin(Theta)

    The weight resolves onto the body X axis through the pitch attitude
    alone. The climb angle does not appear: ``Theta`` is measured from the
    horizon in Figure 8.5, so the tilt of the weight is already accounted
    for, and ``gamma_c`` enters only through the aerodynamic angles upstream.

    ``linearized=True`` puts ``sin(Theta) -> Theta``, which is the row
    Table 8.4 prints as ``-G.W. Theta``, worth -20,000 Theta.

    Scale
    -----
    At 115 knots the five contributions are roughly +536, -40, +10, -56 and
    -778 against a weight term of +330. The fuselage drag is the whole
    equation and the rotor H-force is what balances it; everything else is
    noise at the 1 % level. That is worth knowing before chasing a residual
    of a few pounds through the stabiliser terms.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        one = np.ones(nn)

        for c in 'MTHVF':
            self.add_input(f'X_{c}', shape=(nn,), val=0.0, units='lbf')
        self.add_input('GW', shape=(nn,), val=0.0, units='lbf',
                       desc='gross weight')
        self.add_input('Theta', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage pitch attitude')

        self.add_output('res_X', shape=(nn,), units='lbf',
                        desc='X-equilibrium residual')

        for c in 'MTHVF':
            self.declare_partials('res_X', f'X_{c}', rows=ar, cols=ar, val=one)
        self.declare_partials('res_X', ['GW', 'Theta'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        theta = inputs['Theta']
        tilt = theta if self.options['linearized'] else np.sin(theta)
        outputs['res_X'] = sum(inputs[f'X_{c}'] for c in 'MTHVF') \
            - inputs['GW'] * tilt

    def compute_partials(self, inputs, J):
        theta, GW = inputs['Theta'], inputs['GW']

        if self.options['linearized']:
            J['res_X', 'GW'], J['res_X', 'Theta'] = -theta, -GW
        else:
            J['res_X', 'GW'] = -np.sin(theta)
            J['res_X', 'Theta'] = -GW * np.cos(theta)
