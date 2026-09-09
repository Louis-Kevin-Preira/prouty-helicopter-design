"""R-equilibrium equation, rolling moments in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", Table 8.11 p. 537. Hover form on p. 531.
Moment arms in Table 8.5 p. 523 and Table 8.10 p. 533.
"""

import numpy as np
import openmdao.api as om


class LatREquilibriumComp(om.ExplicitComponent):
    """Sum of the rolling moments about the c.g., Table 8.11 p. 537::

        res_R = R_M + Y_M h_M + Z_M y_M
                    + Y_T h_T
                    + Y_V h_V
              + R_F + Y_F h_F

    Side forces times heights, plus the hub rolling moment and the fuselage
    dihedral effect. Two things differ from the pitching moment equation.

    ``Z_M y_M`` has no counterpart in pitch. It is the main rotor thrust
    acting through a *lateral* c.g. offset, and it is the only place in the
    chapter where lateral loading enters. It is zero for the example
    helicopter, and it is what Table 8.9 p. 532 varies in its third special
    case: with a very stiff hub and any ``y_M``, the flapping goes to zero
    and the fuselage carries the offset by banking instead.

    And there is no ``Z l`` pair here, only ``Y h``. A vertical force
    produces no rolling moment about a longitudinal axis unless it acts
    off the centreline, which is exactly the ``Z_M y_M`` term.

    Scale
    -----
    The hub moment and the ``Y_M h_M`` term dominate, and they act together:
    ``dR_M/db1s = 200,940`` plus ``T_M h_M = 154,545`` make the 355,485
    coefficient of ``b1s_M`` that Table 8.11 prints. As in hover, close to
    half the roll stiffness is the rotor's height above the c.g. rather than
    the hub.
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
            self.add_input(f'h_{c}', val=0.0, units='ft',
                           desc=f'{c} height above the c.g.')
        self.add_input('Z_M', shape=(nn,), val=0.0, units='lbf')
        self.add_input('y_M', val=0.0, units='ft',
                       desc='lateral c.g. offset of the rotor')
        self.add_input('R_M', shape=(nn,), val=0.0, units='lbf*ft',
                       desc='main rotor hub rolling moment')
        self.add_input('R_F', shape=(nn,), val=0.0, units='lbf*ft',
                       desc='fuselage rolling moment')

        self.add_output('res_R', shape=(nn,), units='lbf*ft',
                        desc='R-equilibrium residual')

        for c in self._stations:
            self.declare_partials('res_R', f'Y_{c}', rows=ar, cols=ar)
            self.declare_partials('res_R', f'h_{c}', rows=ar, cols=zeros)
        self.declare_partials('res_R', 'Z_M', rows=ar, cols=ar)
        self.declare_partials('res_R', 'y_M', rows=ar, cols=zeros)
        for name in ('R_M', 'R_F'):
            self.declare_partials('res_R', name, rows=ar, cols=ar, val=one)

    def compute(self, inputs, outputs):
        total = inputs['R_M'] + inputs['R_F'] + inputs['Z_M'] * inputs['y_M'][0]
        for c in self._stations:
            total = total + inputs[f'Y_{c}'] * inputs[f'h_{c}'][0]
        outputs['res_R'] = total

    def compute_partials(self, inputs, J):
        for c in self._stations:
            J['res_R', f'Y_{c}'] = inputs[f'h_{c}'][0]
            J['res_R', f'h_{c}'] = inputs[f'Y_{c}']
        J['res_R', 'Z_M'] = inputs['y_M'][0]
        J['res_R', 'y_M'] = inputs['Z_M']
