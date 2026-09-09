"""Effective gross weight in a manoeuvre.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 529, first step. Anchor in Table 8.7
p. 529.
"""

import numpy as np
import openmdao.api as om


class ManeuverWeightComp(om.ExplicitComponent):
    """Effective gross weight at load factor n, p. 529::

        GW_eff = n GW

    p. 529 evaluates angle of attack stability by trimming in a steady
    descending turn at the same speed and the same collective, and its first
    step is to run the trim "of a helicopter whose effective gross weight is
    equal to its actual gross weight multiplied by the load factor".

    One line, and it earns a component for two reasons. It is the only place
    the load factor enters the longitudinal trim, so naming it makes the
    manoeuvre visible in a model that otherwise looks like level flight. And
    the substitution is an approximation worth being able to point at: the
    turn is replaced by a straight flight at a heavier weight, which captures
    the thrust the rotor must make but not the pitch rate. p. 529 corrects
    for that separately, in its second step, with ``ManeuverRateComp`` and
    ``LongitudinalCyclicTurnComp`` of Chapter 7.

    Anchor
    ------
    Table 8.7 p. 529 runs the example helicopter at 115 knots and ``n = 1.3``
    with collective held at the level flight value, giving 26,000 lb of
    effective weight. The uncorrected cyclic comes out at 11.8 deg against
    8.9 in level flight, and the Chapter 7 pitch rate correction brings it to
    11.4.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('GW', shape=(nn,), val=1.0, units='lbf',
                       desc='actual gross weight')
        self.add_input('n', shape=(nn,), val=1.0, desc='load factor')

        self.add_output('GW_eff', shape=(nn,), units='lbf',
                        desc='effective gross weight for the trim')

        self.declare_partials('GW_eff', ['GW', 'n'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['GW_eff'] = inputs['n'] * inputs['GW']

    def compute_partials(self, inputs, J):
        J['GW_eff', 'GW'] = inputs['n']
        J['GW_eff', 'n'] = inputs['GW']
