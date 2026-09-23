"""Moment about the aircraft centre of gravity due to rotor flapping.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 476 and 479.
"""

import numpy as np
import openmdao.api as om


class CGMomentComp(om.ExplicitComponent):
    """Pitch and roll moments about the centre of gravity (p. 476)::

        M_CG = (dM_M/da_1s) a_1s + (T a_1s + H_{a1s=0}) h_M - T l_M

    Three contributions: the hub couple from rotor stiffness, the in-plane
    rotor force acting a height ``h_M`` above the centre of gravity, and the
    thrust acting at a horizontal offset ``l_M`` from it.

    Options
    -------
    num_nodes : int
    inplane_force : {'simple', 'external'}
        ``'simple'`` uses the ``T a_1s + H_{a1s=0}`` of p. 476, which assumes
        the total rotor vector is perpendicular to the tip path plane.
        ``'external'`` takes the in-plane forces ``H`` and ``Y`` directly, so
        that the inflow correction of p. 479 can be applied upstream.

    Why the external option matters
    -------------------------------
    p. 478 opens by calling the perpendicular assumption satisfactory only
    "for simple analyses". The corrected derivative of p. 479 is::

        d(C_H/sigma)/da_1s = C_T/sigma + (a/8) lambda'

    In dimensional terms that is ``T + (a/8) lambda' sigma rho A (Omega R)^2``,
    and ``lambda'`` is negative, so the second term subtracts. For the example
    helicopter in hover it takes the in-plane force derivative from 20,000 lb
    down to about 9,300 lb, a factor of 2.2. The rotor force contribution to
    the moment falls from 150,000 to 69,400 ft-lb per radian, against the
    73,500 the p. 479 table quotes.

    In other words the ``'simple'`` route overstates the flapping stiffness
    of the aircraft by more than a factor of two in hover. Prouty makes the
    same point on p. 479: the effect "has the effect of reducing the damping
    that one might expect from rotor flapping".

    Sign conventions
    ----------------
    ``M_CG`` positive nose up, ``L_CG`` positive right roll.
    ``h_M`` is the hub height above the centre of gravity.
    ``l_M`` is positive when the centre of gravity is ahead of the shaft, so
    thrust acts behind it and ``-T l_M`` is nose down. ``y_M`` is the lateral
    equivalent, positive with the centre of gravity to the right.

    The roll equation is the symmetric analogue and is not printed in the
    book. p. 479 does confirm the force side of it: the rotor Y-force
    behaves exactly as the H-force, ``d(C_Y/sigma)/db_1s = C_T/sigma +
    (a/8) lambda'``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('inplane_force', values=('simple', 'external'),
                             default='simple')

    def setup(self):
        nn = self.options['num_nodes']
        simple = self.options['inplane_force'] == 'simple'

        self.add_input('M_M', val=np.zeros(nn), units='lbf*ft',
                       desc='Hub pitching moment.')
        self.add_input('L_M', val=np.zeros(nn), units='lbf*ft',
                       desc='Hub rolling moment.')
        self.add_input('T', val=np.zeros(nn), units='lbf',
                       desc='Rotor thrust.')
        self.add_input('h_M', val=0.0, units='ft',
                       desc='Hub height above the centre of gravity.')
        self.add_input('l_M', val=0.0, units='ft',
                       desc='Centre of gravity ahead of the shaft.')
        self.add_input('y_M', val=0.0, units='ft',
                       desc='Centre of gravity right of the shaft.')

        if simple:
            self.add_input('a_1s', val=np.zeros(nn), units='rad',
                           desc='Longitudinal flapping.')
            self.add_input('b_1s', val=np.zeros(nn), units='rad',
                           desc='Lateral flapping.')
            self.add_input('H_0', val=np.zeros(nn), units='lbf',
                           desc='H-force at zero longitudinal flapping.')
            self.add_input('Y_0', val=np.zeros(nn), units='lbf',
                           desc='Y-force at zero lateral flapping.')
            pitch = ['M_M', 'T', 'a_1s', 'H_0']
            roll = ['L_M', 'T', 'b_1s', 'Y_0']
        else:
            self.add_input('H', val=np.zeros(nn), units='lbf',
                           desc='Total in-plane force along x.')
            self.add_input('Y', val=np.zeros(nn), units='lbf',
                           desc='Total in-plane force along y.')
            pitch = ['M_M', 'T', 'H']
            roll = ['L_M', 'T', 'Y']

        self.add_output('M_CG', val=np.zeros(nn), units='lbf*ft',
                        desc='Pitching moment about the c.g., nose up.')
        self.add_output('L_CG', val=np.zeros(nn), units='lbf*ft',
                        desc='Rolling moment about the c.g., right.')

        ar = np.arange(nn)
        self.declare_partials('M_CG', pitch, rows=ar, cols=ar)
        self.declare_partials('L_CG', roll, rows=ar, cols=ar)
        self.declare_partials('M_CG', ['h_M', 'l_M'])
        self.declare_partials('L_CG', ['h_M', 'y_M'])

    def _forces(self, inputs):
        if self.options['inplane_force'] == 'simple':
            return (inputs['T'] * inputs['a_1s'] + inputs['H_0'],
                    inputs['T'] * inputs['b_1s'] + inputs['Y_0'])
        return inputs['H'], inputs['Y']

    def compute(self, inputs, outputs):
        H, Y = self._forces(inputs)
        T, h_M = inputs['T'], inputs['h_M']

        outputs['M_CG'] = inputs['M_M'] + H * h_M - T * inputs['l_M']
        outputs['L_CG'] = inputs['L_M'] + Y * h_M - T * inputs['y_M']

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        H, Y = self._forces(inputs)
        T, h_M = inputs['T'], inputs['h_M']
        ones = np.ones(nn)

        J['M_CG', 'M_M'] = ones
        J['L_CG', 'L_M'] = ones
        J['M_CG', 'h_M'] = H.reshape(-1, 1)
        J['L_CG', 'h_M'] = Y.reshape(-1, 1)
        J['M_CG', 'l_M'] = -T.reshape(-1, 1)
        J['L_CG', 'y_M'] = -T.reshape(-1, 1)

        if self.options['inplane_force'] == 'simple':
            J['M_CG', 'T'] = inputs['a_1s'] * h_M - inputs['l_M']
            J['L_CG', 'T'] = inputs['b_1s'] * h_M - inputs['y_M']
            J['M_CG', 'a_1s'] = T * h_M
            J['L_CG', 'b_1s'] = T * h_M
            J['M_CG', 'H_0'] = ones * h_M
            J['L_CG', 'Y_0'] = ones * h_M
        else:
            J['M_CG', 'T'] = -ones * inputs['l_M']
            J['L_CG', 'T'] = -ones * inputs['y_M']
            J['M_CG', 'H'] = ones * h_M
            J['L_CG', 'Y'] = ones * h_M
