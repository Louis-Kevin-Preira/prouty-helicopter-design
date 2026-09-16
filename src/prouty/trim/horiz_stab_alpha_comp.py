"""Angle of attack of the horizontal stabiliser chord line.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 489, Figure 8.5 p. 488. The identity
relating the two forms is on p. 525. Anchors in Table 8.5 p. 523 and
Table 8.8 p. 530.
"""

import numpy as np
import openmdao.api as om


class HorizStabAlphaComp(om.ExplicitComponent):
    """Angle of attack of the horizontal stabiliser, p. 489.

    p. 489 gives the same angle twice::

        alpha_H = Theta + i_H - (eps_MH + eps_FH) - gamma_c
        alpha_H = alpha_TPP - a1s_M - i_M + i_H - (eps_MH + eps_FH)

    The first measures the airframe from the horizon, the second from the
    tip path plane. They agree because

        alpha_TPP = Theta + a1s_M + i_M - gamma_c

    which p. 525 prints without the ``gamma_c`` because it is writing level
    flight.

    Options
    -------
    num_nodes : int
    reference : {'attitude', 'tpp'}
        Which of the two to evaluate. ``'attitude'`` is the default and is
        what the trim solver wants, since ``Theta`` is one of its three
        unknowns. ``'tpp'`` is for coming the other way, from Chapter 3,
        where ``alpha_TPP`` is a direct output of ``TppAngleComp`` and the
        attitude is not yet known.

    Do not pass gamma_c to the tpp form
    -----------------------------------
    ``'tpp'`` takes no climb angle, and that is not an omission. The
    identity above shows ``alpha_TPP`` is already measured from the flight
    path, so it carries ``gamma_c`` inside it. Subtracting the climb angle
    again is the one way to get this component wrong, and it fails quietly:
    in level flight the two forms agree exactly and nothing shows up until
    someone runs a climb. Table 8.8 p. 530 has ``gamma_c`` reaching 9.7 deg
    in the 2,000 ft/min climb, which is larger than ``alpha_H`` itself at
    that point.

    Anchor
    ------
    At 115 knots, with ``Theta = -0.0165`` from p. 522, ``i_H = -0.052`` from
    Table 8.5, ``eps_MH = 0.0607`` from p. 493 and ``eps_FH = 0.0109`` from
    p. 498::

        alpha_H = -0.1401 rad = -8.03 deg

    against the -7.9 deg Table 8.8 lists for level flight, a gap of 0.13 deg.
    Feeding the same inputs through the tpp form with
    ``alpha_TPP = -0.0355`` and ``a1s_M = -0.019`` returns the identical
    value, as it must.

    Notes
    -----
    ``i_M`` appears only in the tpp form. The attitude form measures from
    the body, which the shaft incidence is defined against, so it cancels.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)
        self.options.declare('reference', values=('attitude', 'tpp'),
                             default='attitude')

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        zeros = np.zeros(nn, dtype=int)
        one, minus = np.ones(nn), -np.ones(nn)

        self.add_input('i_H', val=0.0, units='rad',
                       desc='stabiliser chord line incidence')
        self.add_input('eps_MH', shape=(nn,), val=0.0, units='rad',
                       desc='main rotor downwash at the stabiliser')
        self.add_input('eps_FH', shape=(nn,), val=0.0, units='rad',
                       desc='fuselage downwash at the stabiliser')

        self.add_output('alpha_H', shape=(nn,), units='rad',
                        desc='stabiliser angle of attack')

        self.declare_partials('alpha_H', 'i_H', rows=ar, cols=zeros, val=one)
        for name in ('eps_MH', 'eps_FH'):
            self.declare_partials('alpha_H', name, rows=ar, cols=ar, val=minus)

        if self.options['reference'] == 'attitude':
            self.add_input('Theta', shape=(nn,), val=0.0, units='rad',
                           desc='fuselage pitch attitude')
            self.add_input('gamma_c', shape=(nn,), val=0.0, units='rad',
                           desc='climb angle')
            self.declare_partials('alpha_H', 'Theta', rows=ar, cols=ar,
                                  val=one)
            self.declare_partials('alpha_H', 'gamma_c', rows=ar, cols=ar,
                                  val=minus)
        else:
            self.add_input('alpha_TPP', shape=(nn,), val=0.0, units='rad',
                           desc='tip path plane angle of attack')
            self.add_input('a1s_M', shape=(nn,), val=0.0, units='rad',
                           desc='longitudinal flapping')
            self.add_input('i_M', val=0.0, units='rad',
                           desc='shaft incidence')
            self.declare_partials('alpha_H', 'alpha_TPP', rows=ar, cols=ar,
                                  val=one)
            self.declare_partials('alpha_H', 'a1s_M', rows=ar, cols=ar,
                                  val=minus)
            self.declare_partials('alpha_H', 'i_M', rows=ar, cols=zeros,
                                  val=minus)

    def compute(self, inputs, outputs):
        downwash = inputs['eps_MH'] + inputs['eps_FH']

        if self.options['reference'] == 'attitude':
            body = inputs['Theta'] - inputs['gamma_c']
        else:
            body = inputs['alpha_TPP'] - inputs['a1s_M'] - inputs['i_M'][0]

        outputs['alpha_H'] = body + inputs['i_H'][0] - downwash
