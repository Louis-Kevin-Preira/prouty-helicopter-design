"""Longitudinal cyclic pitch at trim.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 8 "The Helicopter in Trim", p. 522. The sum it splits comes from
Chapter 3, p. 168.
"""

import numpy as np
import openmdao.api as om


class LongCyclicPitchComp(om.ExplicitComponent):
    """Longitudinal cyclic pitch, p. 522::

        B_1 = (B_1 + a1s_M) - a1s_M

    Trivial arithmetic carrying the whole point of the chapter.

    What Chapter 3 could not give you
    ---------------------------------
    The rotor aerodynamics of p. 168 set only the *sums* ``B_1 + a1s`` and
    ``A_1 - b1s``, not the two terms separately: tilting the control plane
    forward by one degree and letting the blades flap back by one degree
    leave the tip path plane, and therefore every force the rotor makes, in
    the same place. ``ClosedFormRotorGroup`` says as much in its own
    docstring. Performance analysis never needs the split, which is why
    Chapter 3 stops at the sum.

    Trim does need it, because the pilot moves the stick and not the tip
    path plane. Splitting the sum requires knowing how far the blades flap
    with respect to the *shaft*, and that is set by the pitching moment
    equilibrium — the equation p. 516 says Chapter 3 ignored by assuming the
    tip path plane stayed perpendicular to the mast. So ``a1s_M`` comes out
    of the M equation of Table 8.4, and this component is where the two
    chapters meet.

    Anchor
    ------
    p. 522, example helicopter at 115 knots: ``(B_1 + a1s_M) = 7.8 deg`` from
    the performance charts of Chapter 3, ``a1s_M = -1.1 deg`` from the trim
    solution, so ``B_1 = 7.8 - (-1.1) = 8.9 deg``.

    Note the direction. The flapping is *aft* and the cyclic is therefore
    further forward than the sum alone suggests, by the whole of ``a1s_M``.
    Reading the trim requirement off the Chapter 3 sum would understate the
    forward stick by 1.1 degrees out of 8.9, which is 12 % of the control
    travel being used.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)

        self.add_input('B1_a1s', shape=(nn,), val=0.0, units='rad',
                       desc='B_1 + a1s, longitudinal control plane tilt')
        self.add_input('a1s_M', shape=(nn,), val=0.0, units='rad',
                       desc='longitudinal flapping wrt the shaft')

        self.add_output('B_1', shape=(nn,), units='rad',
                        desc='longitudinal cyclic pitch')

        self.declare_partials('B_1', 'B1_a1s', rows=ar, cols=ar,
                              val=np.ones(nn))
        self.declare_partials('B_1', 'a1s_M', rows=ar, cols=ar,
                              val=-np.ones(nn))

    def compute(self, inputs, outputs):
        outputs['B_1'] = inputs['B1_a1s'] - inputs['a1s_M']
