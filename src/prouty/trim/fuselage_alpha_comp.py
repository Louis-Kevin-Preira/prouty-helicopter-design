"""Angle of attack of the fuselage.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 513, Figure 8.23 p. 512. Equivalent
to the Chapter 3 form of p. 192. Anchor in Table 8.8 p. 530.
"""

import numpy as np
import openmdao.api as om


class FuselageAlphaComp(om.ExplicitComponent):
    """Fuselage angle of attack, p. 513::

        alpha_F = Theta - gamma_c - eps_MF

    The fuselage does not see the free stream: the rotor turns it downward
    by ``eps_MF``, and the airframe is pitched by ``Theta`` against a flight
    path inclined by ``gamma_c``.

    Same angle as Chapter 3
    -----------------------
    ``FuselageAngleComp`` of Chapter 3, p. 192, computes the same quantity
    from the tip path plane instead::

        alpha_F = lambda'/mu - i_s - a1s = alpha_TPP - i_M - a1s - v1/V

    Substituting ``alpha_TPP = Theta + a1s + i_M - gamma_c`` from p. 525
    leaves ``alpha_F = Theta - gamma_c - v1/V``, which is this equation with
    ``eps_MF = v1/V``. That is exactly what ``RotorDownwashComp`` returns at
    ``v_ratio = 1``, and 1.0 is what Table 8.4 uses at the fuselage. The two
    chapters agree identically, by two different routes.

    The Chapter 3 form is singular at ``mu = 0`` because it divides by the
    tip speed ratio. This one is not: in hover it returns
    ``Theta - eps_MF``, which is finite and meaningless in the same way the
    fuselage angle of attack itself is. Neither form should be trusted near
    hover; only this one fails quietly, so the longitudinal hover solution
    of p. 516 neglects the airframe aerodynamics outright rather than
    evaluating them at an angle it cannot define.

    Anchor
    ------
    At 115 knots with ``Theta = -0.0165``, ``gamma_c = 0`` and
    ``eps_MF = T_M/(4 q A_M) = 0.0404``, this gives
    ``alpha_F = -0.0569 rad = -3.26 deg``, against the -3.3 deg Table 8.8
    p. 530 lists for level flight.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        one, minus = np.ones(nn), -np.ones(nn)

        self.add_input('Theta', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage pitch attitude')
        self.add_input('gamma_c', shape=(nn,), val=0.0, units='rad',
                       desc='climb angle')
        self.add_input('eps_MF', shape=(nn,), val=0.0, units='rad',
                       desc='main rotor downwash at the fuselage')

        self.add_output('alpha_F', shape=(nn,), units='rad',
                        desc='fuselage angle of attack')

        self.declare_partials('alpha_F', 'Theta', rows=ar, cols=ar, val=one)
        for name in ('gamma_c', 'eps_MF'):
            self.declare_partials('alpha_F', name, rows=ar, cols=ar, val=minus)

    def compute(self, inputs, outputs):
        outputs['alpha_F'] = (inputs['Theta'] - inputs['gamma_c']
                              - inputs['eps_MF'])
