"""Hub moments produced by first-harmonic flapping.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 7 "Rotor Flapping Characteristics", pp. 476-477.
"""

import numpy as np
import openmdao.api as om


class HubMomentComp(om.ExplicitComponent):
    """Pitch and roll moments at the hub, from the rotor stiffness.

    p. 477 builds the pitching moment by averaging the vertical component of
    blade centrifugal force over azimuth::

        M_M = e b Omega^2 a_1s (M_b/g) (1/2pi) int_0^2pi cos^2(psi) dpsi
            = (1/2) e b Omega^2 a_1s (M_b/g)

    which is ``(dM_M/da_1s) a_1s``. The code multiplies the stiffness from
    ``RotorStiffnessComp`` rather than rebuilding it, so it inherits whichever
    static-moment convention was selected there (entry C7-6).

    The roll moment is the same derivation with ``beta - a_0 = -b_1s
    sin(psi)`` in place of ``-a_1s cos(psi)``. Prouty does not write it out,
    but ``<sin^2> = <cos^2> = 1/2``, so::

        L_M = (dM_M/da_1s) b_1s

    Sign convention as in Figures 7.9 and 7.11: the hub moment follows the
    tip path plane, so positive ``a_1s`` (disc back) gives a nose-up moment
    and positive ``b_1s`` (disc down to the right) gives a right-roll moment.

    Two-bladed rotors
    -----------------
    The ``b/2`` factor comes from the azimuthal average and is only valid for
    three blades or more. Summing ``cos^2(psi + 2 pi k / b)`` over the blades
    gives a constant ``b/2`` for ``b >= 3``, but ``1 + cos(2 psi)`` for
    ``b = 2``: a two-bladed rotor has no steady hub moment of this kind, it
    has a 2/rev oscillation between zero and twice the value returned here.
    Teetering rotors avoid the question by having no offset at all.

    Notes
    -----
    Only the first harmonic contributes. Coning produces no hub moment,
    because its centrifugal force has no once-per-rev component.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('dMM_da1s', val=np.zeros(nn), units='lbf*ft/rad',
                       desc='Rotor stiffness.')
        self.add_input('a_1s', val=np.zeros(nn), units='rad',
                       desc='Longitudinal flapping.')
        self.add_input('b_1s', val=np.zeros(nn), units='rad',
                       desc='Lateral flapping.')

        self.add_output('M_M', val=np.zeros(nn), units='lbf*ft',
                        desc='Hub pitching moment, positive nose up.')
        self.add_output('L_M', val=np.zeros(nn), units='lbf*ft',
                        desc='Hub rolling moment, positive right.')

        ar = np.arange(nn)
        self.declare_partials('M_M', ['dMM_da1s', 'a_1s'], rows=ar, cols=ar)
        self.declare_partials('L_M', ['dMM_da1s', 'b_1s'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        outputs['M_M'] = inputs['dMM_da1s'] * inputs['a_1s']
        outputs['L_M'] = inputs['dMM_da1s'] * inputs['b_1s']

    def compute_partials(self, inputs, J):
        J['M_M', 'dMM_da1s'] = inputs['a_1s']
        J['M_M', 'a_1s'] = inputs['dMM_da1s']
        J['L_M', 'dMM_da1s'] = inputs['b_1s']
        J['L_M', 'b_1s'] = inputs['dMM_da1s']
