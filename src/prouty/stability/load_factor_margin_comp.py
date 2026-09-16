"""Angle-of-attack stability as the load factor rises.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", p. 622 and Figure 9.18, p. 623.
"""

import warnings

import numpy as np
import openmdao.api as om

#: Figure 9.18, p. 623, read off its endpoints: the line runs from
#: (1.0, -214) to (1.82, 0), so +261 ft lb/(ft/sec) per g on a constant -475.
#: Chapter 9 does not print the split, only the drawn line.
FIGURE_9_18 = {'rotor': 261.0, 'airframe': -475.0, 'neutral': 1.82}


class LoadFactorMarginComp(om.ExplicitComponent):
    """How much manoeuvre an aircraft has before it loses angle-of-attack
    stability.

    Every other analysis in Chapter 9 is at one g. p. 622 says why that is not
    enough::

        a helicopter that is stable in level flight will probably be unstable
        at some higher load factor at the same speed. This is in contrast to
        an airplane, whose angle-of-attack stability is nearly invariant with
        wing angle of attack. The difference is due to the contribution of the
        rearward tilt of the thrust vector; the higher the thrust, the
        stronger the destabilizing moment due to nose-up flapping. The
        airframe, on the other hand, maintains a constant stabilizing
        influence.

    So the derivative splits in two and only one half moves::

        dM/dzdot(n) = n (rotor contribution) + (airframe contribution)

    with the rotor part positive and destabilizing and the airframe part
    negative and constant. The aircraft is stable in pitch to angle of attack
    while the sum is negative, and Figure 9.18 is that straight line.

    For the example helicopter with the 90 ft^2 stabilizer -- the one size
    p. 622 says "would be enough to provide a margin of positive
    angle-of-attack stability in level flight at 115 knots" -- the margin is
    **gone at 1.82 g**, a turn at 56 degrees of bank or a moderate pull-up.
    p. 622: below that "the helicopter, upon encountering an up-gust, would
    pitch down by itself"; above it "the rotor would overpower the stabilizer
    and would pitch the helicopter nose-up unless prevented by an alert
    pilot".

    Constrain ``dM_dzdot_at_limit``, not the neutral point
    ------------------------------------------------------
    ``load_factor_at_neutral`` is ``-airframe/rotor`` and runs to infinity as
    the rotor contribution vanishes, which makes it a poor thing to put in a
    gradient. ``dM_dzdot_at_limit`` is the derivative evaluated at whatever
    limit load factor the aircraft is being designed to, is finite
    everywhere, and asks the question a designer actually has: *is it still
    stable at 2 g?*

    Options
    -------
    num_nodes : int

    Example helicopter, 90 ft^2 stabilizer
    --------------------------------------
    ================================ ========== ===========
                                     model      Figure 9.18
    ================================ ========== ===========
    dM/dzdot at 1 g                  -214       -214
    load factor at neutral           1.820      1.82
    ================================ ========== ===========

    The **total** at one g agrees with the derivative tables to 5 %: summing
    Table 9.16's columns with the stabilizer counted five times gives
    ``495 - 1,095 + 374 = -226`` against the figure's -214.

    The **split** cannot be recovered from Table 9.8. Its three terms for
    ``dM/dzdot`` are the hub spring (333), the ``h_M`` arm (45) and the
    ``l_M`` arm (131); taking the last two as the thrust-dependent pair gives
    176 where the figure needs 261. Chapter 9 never prints the split, so
    ``FIGURE_9_18`` holds the values read back off the line, and they are
    inputs. This is the second place in the chapter where a number comes from
    a figure rather than an equation.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('dM_dzdot_rotor', units='lbf*s',
                       val=np.full(nn, FIGURE_9_18['rotor']),
                       desc='Part that scales with load factor.')
        self.add_input('dM_dzdot_airframe', units='lbf*s',
                       val=np.full(nn, FIGURE_9_18['airframe']),
                       desc='Part that does not.')
        self.add_input('load_factor', val=np.ones(nn),
                       desc='Flight condition the aircraft is trimmed at.')
        self.add_input('limit_load_factor', val=np.full(nn, 2.0),
                       desc='What it is being designed to hold.')

        self.add_output('dM_dzdot', val=np.zeros(nn), units='lbf*s')
        self.add_output('dM_dzdot_at_limit', val=np.zeros(nn), units='lbf*s',
                        desc='The quantity to constrain.')
        self.add_output('load_factor_at_neutral', val=np.zeros(nn),
                        desc='Where stability vanishes; infinite if it never does.')
        self.add_output('load_factor_margin', val=np.zeros(nn))

        self.declare_partials('dM_dzdot',
                              ['dM_dzdot_rotor', 'dM_dzdot_airframe',
                               'load_factor'], rows=ar, cols=ar)
        self.declare_partials('dM_dzdot_at_limit',
                              ['dM_dzdot_rotor', 'dM_dzdot_airframe',
                               'limit_load_factor'], rows=ar, cols=ar)
        for name in ('load_factor_at_neutral', 'load_factor_margin'):
            self.declare_partials(name, ['dM_dzdot_rotor',
                                         'dM_dzdot_airframe'],
                                  rows=ar, cols=ar)
        self.declare_partials('load_factor_margin', 'load_factor',
                              rows=ar, cols=ar, val=-1.0)

    def compute(self, inputs, outputs):
        rotor, airframe = inputs['dM_dzdot_rotor'], inputs['dM_dzdot_airframe']

        outputs['dM_dzdot'] = rotor * inputs['load_factor'] + airframe
        outputs['dM_dzdot_at_limit'] = (rotor * inputs['limit_load_factor']
                                        + airframe)

        destabilising = np.real(rotor) > 0.0
        if not np.all(destabilising):
            warnings.warn('the rotor contribution to dM/dzdot is not '
                          'destabilising at every node: angle-of-attack '
                          'stability never vanishes there, and the neutral '
                          'load factor is set to zero.')
        safe = np.where(destabilising, rotor, 1.0)
        neutral = np.where(destabilising, -airframe / safe, 0.0)

        outputs['load_factor_at_neutral'] = neutral
        outputs['load_factor_margin'] = neutral - inputs['load_factor']

    def compute_partials(self, inputs, J):
        rotor, airframe = inputs['dM_dzdot_rotor'], inputs['dM_dzdot_airframe']
        one = np.ones_like(rotor)

        J['dM_dzdot', 'dM_dzdot_rotor'] = inputs['load_factor']
        J['dM_dzdot', 'dM_dzdot_airframe'] = one
        J['dM_dzdot', 'load_factor'] = rotor

        J['dM_dzdot_at_limit', 'dM_dzdot_rotor'] = inputs['limit_load_factor']
        J['dM_dzdot_at_limit', 'dM_dzdot_airframe'] = one
        J['dM_dzdot_at_limit', 'limit_load_factor'] = rotor

        destabilising = rotor > 0.0
        safe = np.where(destabilising, rotor, 1.0)
        d_rotor = np.where(destabilising, airframe / safe ** 2, 0.0)
        d_airframe = np.where(destabilising, -1.0 / safe, 0.0)

        for name in ('load_factor_at_neutral', 'load_factor_margin'):
            J[name, 'dM_dzdot_rotor'] = d_rotor
            J[name, 'dM_dzdot_airframe'] = d_airframe
