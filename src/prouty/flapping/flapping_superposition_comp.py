"""Superposition of steady flapping and flapping due to angular rates.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 473-474.
"""

import numpy as np
import openmdao.api as om


class FlappingSuperpositionComp(om.ExplicitComponent):
    """Add the rate-induced flapping to the steady flapping (p. 473)::

        a_1s,total = a_1s + a_1s,rate
        b_1s,total = b_1s + b_1s,rate

    Why this is exact
    -----------------
    Prouty writes only that the rate terms "can be added to the previously
    derived equations for flapping in steady flight". The addition is not an
    approximation: both problems reduce to the *same* 2x2 system (see
    ``flapping_2x2``), differing only in the right-hand side, and a linear
    system solved with the sum of two right-hand sides gives the sum of the
    two solutions. Verified symbolically.

    What it does not cover
    ----------------------
    Coning is assumed unchanged. That is not true in a real manoeuvre: a
    pull-up raises the thrust, hence the coning, which tilts the disc down to
    the right and partly cancels the left tilt from pitch rate (p. 474).
    Prouty handles that separately as a load-factor effect on the lateral
    cyclic, ``Delta_A_1,manoeuvre = (n - 1) A_1,level``, not through this
    superposition. Feeding a manoeuvre value of ``C_T/sigma`` into the steady
    group is the way to capture it here.

    A third source, the sideslip that builds up if the pedals are held fixed
    during a pull-up, is the rotor dihedral effect of p. 474 and is outside
    Chapter 7.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('a_1s', val=np.zeros(nn), units='rad',
                       desc='Longitudinal flapping in steady flight.')
        self.add_input('b_1s', val=np.zeros(nn), units='rad',
                       desc='Lateral flapping in steady flight.')
        self.add_input('a_1s_rate', val=np.zeros(nn), units='rad',
                       desc='Longitudinal flapping due to rates.')
        self.add_input('b_1s_rate', val=np.zeros(nn), units='rad',
                       desc='Lateral flapping due to rates.')

        self.add_output('a_1s_total', val=np.zeros(nn), units='rad',
                        desc='Total longitudinal flapping.')
        self.add_output('b_1s_total', val=np.zeros(nn), units='rad',
                        desc='Total lateral flapping.')

        ar = np.arange(nn)
        for out, sources in (('a_1s_total', ('a_1s', 'a_1s_rate')),
                             ('b_1s_total', ('b_1s', 'b_1s_rate'))):
            self.declare_partials(out, sources, rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        outputs['a_1s_total'] = inputs['a_1s'] + inputs['a_1s_rate']
        outputs['b_1s_total'] = inputs['b_1s'] + inputs['b_1s_rate']
