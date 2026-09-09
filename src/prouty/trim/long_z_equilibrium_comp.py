"""Z-equilibrium equation in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", Table 8.4 p. 519. Hover form on p. 516.
"""

import numpy as np
import openmdao.api as om


class LongZEquilibriumComp(om.ExplicitComponent):
    """Sum of the Z forces, Table 8.4 p. 519::

        res_Z = Z_M + Z_T + Z_H + Z_V + Z_F + G.W. cos(Theta)

    ``linearized=True`` puts ``cos(Theta) -> 1``, which is the row Table 8.4
    prints as ``+ G.W.``, worth 20,000. At the 1.4 degrees of pitch this
    helicopter trims to, the cosine is 0.9997, so the two forms differ by
    6 lb — the least consequential linearisation in the chapter, and the one
    that makes the equation return ``T_M`` almost directly.

    Scale
    -----
    ``Z_M = -T_M cos(a1s_M + i_M)`` is the equation. Everything else totals
    about 425 lb against 20,556: the stabiliser download of 270, the
    fuselage lift of 260, and 5 lb from the fin. That is why p. 517 can say
    the second equation gives a unique thrust, and why the hover form solves
    for ``T_M`` in one line.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        one = np.ones(nn)

        for c in 'MTHVF':
            self.add_input(f'Z_{c}', shape=(nn,), val=0.0, units='lbf')
        self.add_input('GW', shape=(nn,), val=0.0, units='lbf',
                       desc='gross weight')
        self.add_input('Theta', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage pitch attitude')

        self.add_output('res_Z', shape=(nn,), units='lbf',
                        desc='Z-equilibrium residual')

        for c in 'MTHVF':
            self.declare_partials('res_Z', f'Z_{c}', rows=ar, cols=ar, val=one)
        self.declare_partials('res_Z', 'GW', rows=ar, cols=ar)
        if not self.options['linearized']:
            self.declare_partials('res_Z', 'Theta', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        lift = 1.0 if self.options['linearized'] else np.cos(inputs['Theta'])
        outputs['res_Z'] = sum(inputs[f'Z_{c}'] for c in 'MTHVF') \
            + inputs['GW'] * lift

    def compute_partials(self, inputs, J):
        if self.options['linearized']:
            J['res_Z', 'GW'] = np.ones_like(inputs['GW'])
        else:
            J['res_Z', 'GW'] = np.cos(inputs['Theta'])
            J['res_Z', 'Theta'] = -inputs['GW'] * np.sin(inputs['Theta'])
