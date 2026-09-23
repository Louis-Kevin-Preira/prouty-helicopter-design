"""The lateral-directional equations of motion in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis". The symbolic matrix is Table 9.19,
pp. 614-615, lower-right submatrix; the numbers are Table 9.20, p. 614; the
characteristic equation and its roots are p. 628.
"""

import numpy as np
import openmdao.api as om

#: (row, column, power of s) -> input name, for the entries that are a single
#: derivative. The rest are assembled in compute.
ENTRIES = {
    (0, 0, 1): 'dY_dydot',
    (1, 0, 1): 'dR_dydot', (1, 1, 1): 'dR_dp', (1, 2, 1): 'dR_dr',
    (2, 0, 1): 'dN_dydot', (2, 1, 1): 'dN_dp', (2, 2, 1): 'dN_dr',
}

DERIVATIVE_UNITS = {
    'dY_dydot': 'lbf*s/ft', 'dY_dp': 'lbf*s/rad', 'dY_dr': 'lbf*s/rad',
    'dR_dydot': 'lbf*s', 'dR_dp': 'lbf*ft*s/rad', 'dR_dr': 'lbf*ft*s/rad',
    'dN_dydot': 'lbf*s', 'dN_dp': 'lbf*ft*s/rad', 'dN_dr': 'lbf*ft*s/rad',
}

#: Table 9.20, p. 614, lateral-directional block, for the tests.
TABLE_9_20_LATERAL = (
    ('-621 s^2 - 107 s', '-3964 s + 20000', '-119,220 s'),
    ('-382 s', '-5000 s^2 - 33,738 s', '6,912 s'),
    ('1207 s', '6,909 s', '-35,000 s^2 - 53,913 s'),
)


class LateralMatrixFFComp(om.ExplicitComponent):
    """The lateral-directional submatrix of Table 9.19, in displacement form.

    States ``y(s)``, ``Phi(s)``, ``Psi(s)``::

        | dY/dydot s - m s^2   (dY/dp + m V Theta)s + G.W.   (dY/dr - m V)s |
        | dR/dydot s           -I_xx s^2 + dR/dp s           dR/dr s        |
        | dN/dydot s           dN/dp s              -I_zz s^2 + dN/dr s     |

    with ``m = G.W./g``. Feed it to ``PolyDeterminantComp(n=3, degree=2,
    n_zero_roots=2)``.

    The gravity term is +G.W.
    ------------------------
    Same sign as in the hover lateral matrix and opposite to the longitudinal
    one, for the same reason: with ``x`` forward, ``y`` right and ``z`` down,
    a right-wing-down ``Phi`` puts ``+W sin(Phi)`` along body ``y`` while a
    nose-up ``Theta`` puts ``-W sin(Theta)`` along body ``x``. Table 9.20
    prints ``-3964 s + 20000`` here against ``3927 s - 20000`` in the X row.

    The two kinematic terms mirror the longitudinal ones
    ----------------------------------------------------
    ``(dY/dp + m V Theta_bar)`` is the roll-rate analogue of the X row's
    ``(dX/dq - m V Theta_bar)``, and it carries the **same** 1,990 with the
    opposite sign: -1,974 of aerodynamic ``dY/dp`` against -1,990 of
    kinematics, giving Table 9.20's -3,964. The two entries are a check on
    each other, and both need ``Theta_bar = -.0165 rad``.

    ``(dY/dr - m V)`` is the centrifugal term, and like its longitudinal
    counterpart it swamps the aerodynamics: -120,555 against +1,397, so
    ``dY/dr`` is 1 % of the entry.

    No product of inertia
    ---------------------
    Table 9.19 carries no ``I_xz``. Roll and yaw are coupled only through the
    aerodynamic derivatives ``dR/dr`` and ``dN/dp``, which for the example
    helicopter are 6,912 and 6,909 -- nearly equal, and both from the tail
    rotor and the vertical stabilizer carried on their arms.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    ``s^4 + 8.460 s^3 + 17.68 s^2 + 45.54 s + 2.2548`` (p. 628), with roots
    ``-6.842``, ``-.7841 +/- 2.4317i`` and ``-.05058``: a fast roll
    convergence, a well damped Dutch roll of 2.6 seconds, and a slow spiral
    convergence halving in about 14 seconds.

    **The lateral-directional motion is stable** where the longitudinal is a
    pure divergence. All four roots are in the left half plane.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        shape = (nn, 3, 3, 3)

        for name, units in DERIVATIVE_UNITS.items():
            self.add_input(name, val=np.zeros(nn), units=units)
        self.add_input('G_W', val=np.full(nn, 20000.0), units='lbf')
        self.add_input('I_xx', val=np.full(nn, 5000.0), units='slug*ft**2')
        self.add_input('I_zz', val=np.full(nn, 35000.0), units='slug*ft**2')
        self.add_input('g', val=np.full(nn, 32.2), units='ft/s**2')
        self.add_input('V', val=np.full(nn, 194.1), units='ft/s')
        self.add_input('Theta_bar', val=np.full(nn, -0.016507), units='rad',
                       desc='Trim pitch attitude.')

        self.add_output('matrix_coeffs', shape=shape)

        def flat(row, col, power):
            return np.ravel_multi_index(
                (ar, np.full(nn, row), np.full(nn, col), np.full(nn, power)),
                shape)

        for (row, col, power), name in ENTRIES.items():
            self.declare_partials('matrix_coeffs', name,
                                  rows=flat(row, col, power), cols=ar, val=1.0)
        self.declare_partials('matrix_coeffs', 'dY_dp', rows=flat(0, 1, 1),
                              cols=ar, val=1.0)
        self.declare_partials('matrix_coeffs', 'dY_dr', rows=flat(0, 2, 1),
                              cols=ar, val=1.0)
        self.declare_partials('matrix_coeffs', 'I_xx', rows=flat(1, 1, 2),
                              cols=ar, val=-1.0)
        self.declare_partials('matrix_coeffs', 'I_zz', rows=flat(2, 2, 2),
                              cols=ar, val=-1.0)

        self.declare_partials('matrix_coeffs', ['G_W', 'g'],
                              rows=np.concatenate([flat(0, 0, 2),
                                                   flat(0, 1, 1),
                                                   flat(0, 2, 1),
                                                   flat(0, 1, 0)]),
                              cols=np.tile(ar, 4))
        self.declare_partials('matrix_coeffs', ['V', 'Theta_bar'],
                              rows=np.concatenate([flat(0, 1, 1),
                                                   flat(0, 2, 1)]),
                              cols=np.tile(ar, 2))

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        mat = np.zeros((nn, 3, 3, 3), dtype=inputs._get_data().dtype)
        mass = inputs['G_W'] / inputs['g']
        momentum = mass * inputs['V']

        for (row, col, power), name in ENTRIES.items():
            mat[:, row, col, power] = inputs[name]

        mat[:, 0, 0, 2] = -mass
        mat[:, 0, 1, 1] = inputs['dY_dp'] + momentum * inputs['Theta_bar']
        mat[:, 0, 1, 0] = inputs['G_W']             # +G.W., see the docstring
        mat[:, 0, 2, 1] = inputs['dY_dr'] - momentum
        mat[:, 1, 1, 2] = -inputs['I_xx']
        mat[:, 2, 2, 2] = -inputs['I_zz']

        outputs['matrix_coeffs'] = mat

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        G_W, g, V = inputs['G_W'], inputs['g'], inputs['V']
        theta, mass = inputs['Theta_bar'], inputs['G_W'] / inputs['g']
        zero, one = np.zeros(nn), np.ones(nn)

        # order: (0,0,2) -m, (0,1,1) +mV theta, (0,2,1) -mV, (0,1,0) +G.W.
        J['matrix_coeffs', 'G_W'] = np.concatenate(
            [-one / g, V * theta / g, -V / g, one])
        J['matrix_coeffs', 'g'] = np.concatenate(
            [mass / g, -mass * V * theta / g, mass * V / g, zero])
        J['matrix_coeffs', 'V'] = np.concatenate([mass * theta, -mass * one])
        J['matrix_coeffs', 'Theta_bar'] = np.concatenate([mass * V, zero])
