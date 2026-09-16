"""N-equilibrium equation, yawing moments in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", Table 8.11 p. 537. Hover form on p. 531.
"""

import numpy as np
import openmdao.api as om


class LatNEquilibriumComp(om.ExplicitComponent):
    """Sum of the yawing moments about the c.g., Table 8.11 p. 537::

        res_N = N_M - Y_M l_M - Y_T l_T - Y_V l_V - Y_F l_F + N_F

    Side forces times lever arms, minus, plus the main rotor torque and the
    fuselage weathercock moment. Nothing else: the X forces produce no
    yawing moment, having no lateral offset in this model.

    In hover this reduces to ``Q_M - l_T T_T = 0``, p. 531, which is the
    antitorque relation ``TailRotorForcesComp`` uses in its ``'antitorque'``
    mode. In forward flight the fin takes a share, and the ``-Y_V l_V`` term
    is what makes the tail rotor thrust drop from 1,540 lb in hover to
    661 lb at 115 knots.

    Scale
    -----
    ``N_M = Q_M = 34,726`` against ``-Y_T l_T = -37 T_T``. The equation is
    a torque balance with corrections, and its coefficient of ``beta``,
    +69,973, is the directional stability: 101,290 of it from the fin,
    -36,900 from the fuselage weathercocking the wrong way. The fin is not
    a trim aid here, it is what makes the sign come out right.
    """

    _stations = 'MTVF'

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('linearized', types=bool, default=False)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        one = np.ones(nn)

        for c in self._stations:
            self.add_input(f'Y_{c}', shape=(nn,), val=0.0, units='lbf')
            self.add_input(f'l_{c}', val=0.0, units='ft',
                           desc=f'{c} offset aft of the c.g.')
        self.add_input('N_M', shape=(nn,), val=0.0, units='lbf*ft',
                       desc='main rotor torque about the yaw axis')
        self.add_input('N_F', shape=(nn,), val=0.0, units='lbf*ft',
                       desc='fuselage yawing moment')

        self.add_output('res_N', shape=(nn,), units='lbf*ft',
                        desc='N-equilibrium residual')

        for c in self._stations:
            self.declare_partials('res_N', f'Y_{c}', rows=ar, cols=ar)
            self.declare_partials('res_N', f'l_{c}', rows=ar, cols=zeros)
        for name in ('N_M', 'N_F'):
            self.declare_partials('res_N', name, rows=ar, cols=ar, val=one)

    def compute(self, inputs, outputs):
        total = inputs['N_M'] + inputs['N_F']
        for c in self._stations:
            total = total - inputs[f'Y_{c}'] * inputs[f'l_{c}'][0]
        outputs['res_N'] = total

    def compute_partials(self, inputs, J):
        for c in self._stations:
            J['res_N', f'Y_{c}'] = -inputs[f'l_{c}'][0]
            J['res_N', f'l_{c}'] = -inputs[f'Y_{c}']
