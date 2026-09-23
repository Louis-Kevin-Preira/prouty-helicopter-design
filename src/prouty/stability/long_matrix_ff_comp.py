"""The longitudinal equations of motion in forward flight, as a matrix.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis". The symbolic matrix is Table 9.19,
pp. 614-615, upper-left submatrix; the numbers are Table 9.20, p. 614; the
characteristic equation and its roots are p. 617.
"""

import numpy as np
import openmdao.api as om

#: (row, column, power of s) -> input name, for the entries that are a single
#: derivative. The rest are assembled in compute.
ENTRIES = {
    (0, 0, 1): 'dX_dxdot',
    (0, 1, 1): 'dX_dzdot',
    (1, 0, 1): 'dZ_dxdot',
    (1, 1, 1): 'dZ_dzdot',
    (1, 1, 2): 'dZ_dzddot',
    (2, 0, 1): 'dM_dxdot',
    (2, 1, 1): 'dM_dzdot',
    (2, 1, 2): 'dM_dzddot',
    (2, 2, 1): 'dM_dq',
}

DERIVATIVE_UNITS = {
    'dX_dxdot': 'lbf*s/ft', 'dX_dzdot': 'lbf*s/ft', 'dX_dq': 'lbf*s/rad',
    'dZ_dxdot': 'lbf*s/ft', 'dZ_dzdot': 'lbf*s/ft',
    'dZ_dzddot': 'lbf*s**2/ft', 'dZ_dq': 'lbf*s/rad',
    'dM_dxdot': 'lbf*s', 'dM_dzdot': 'lbf*s', 'dM_dzddot': 'lbf*s**2',
    'dM_dq': 'lbf*ft*s/rad',
}

#: Table 9.20, p. 614, longitudinal block, for the docstring and the tests.
TABLE_9_20_LONGITUDINAL = (
    ('-621 s^2 - 20 s', '-8 s', '3927 s - 20000'),
    ('49 s', '-621 s^2 - 287 s', '120,400 s'),
    ('144 s', '9 s^2 + 650 s', '-40,000 s^2 - 43,752 s'),
)


class LongMatrixFFComp(om.ExplicitComponent):
    """The longitudinal submatrix of Table 9.19, in displacement form.

    p. 616 explains the arrangement: the six equations of motion are written
    so the longitudinal three form the upper-left submatrix, the
    lateral-directional three the lower right, and the other two corners are
    the cross-coupling. This component builds the upper-left corner::

        | dX/dxdot s - m s^2      dX/dzdot s        (dX/dq - m V Theta) s - G.W. |
        | dZ/dxdot s              (dZ/dzddot - m)s^2 + dZ/dzdot s   (dZ/dq + m V) s |
        | dM/dxdot s              dM/dzddot s^2 + dM/dzdot s   -I_yy s^2 + dM/dq s |

    with ``m = G.W./g``. Feed it to ``PolyDeterminantComp(n=3, degree=2,
    n_zero_roots=2)``.

    Displacement form, not velocity form
    ------------------------------------
    Unlike the hover matrix of p. 597, every entry here carries an extra
    ``s``: the states are ``x(s)``, ``z(s)``, ``Theta(s)`` rather than
    ``xdot``, ``zdot``, ``Theta``. The determinant is therefore degree six and
    carries an ``s^2`` rigid-body factor, which ``n_zero_roots=2`` divides out
    to leave the quartic of p. 617.

    Two terms forward flight adds
    -----------------------------
    Both come from the trim velocity and neither exists in hover.

    ``(dX/dq - m V Theta_bar)`` in the X row: a pitch rate in forward flight
    rotates the velocity vector, and the trim pitch attitude decides how much
    of that lands on the X axis. For the example helicopter it is the larger
    part of the entry -- ``1937`` of aerodynamic ``dX/dq`` against ``+1990``
    of kinematics, giving the ``3927`` Table 9.20 prints.

    ``(dZ/dq + m V)`` in the Z row is the centrifugal term: a pitch rate at
    194 ft/sec throws 120,555 lb of normal force against the ``-217`` of
    aerodynamic ``dZ/dq``. The aerodynamics are 0.2 % of that entry.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    Fed the Table 9.16 totals with ``G.W. = 20,000 lb``,
    ``I_yy = 40,000 slug ft^2`` and ``Theta_bar = -.0165 rad``, every entry
    reproduces Table 9.20's longitudinal block, and the determinant gives
    ``s^4 + 1.545 s^3 - 2.618 s^2 + .0228 s + .0949`` against the printed
    quartic of p. 617.

    ``Theta_bar`` is not printed in Chapter 9. It is recovered from Table
    9.20's own ``3927 s`` entry, which needs ``-0.95 deg`` -- a slightly
    nose-down attitude at 115 knots, as expected.
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
        self.add_input('I_yy', val=np.full(nn, 40000.0), units='slug*ft**2')
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
        self.declare_partials('matrix_coeffs', 'dX_dq', rows=flat(0, 2, 1),
                              cols=ar, val=1.0)
        self.declare_partials('matrix_coeffs', 'dZ_dq', rows=flat(1, 2, 1),
                              cols=ar, val=1.0)
        self.declare_partials('matrix_coeffs', 'I_yy', rows=flat(2, 2, 2),
                              cols=ar, val=-1.0)

        self._mass = np.concatenate([flat(0, 0, 2), flat(1, 1, 2),
                                     flat(0, 2, 1), flat(1, 2, 1),
                                     flat(0, 2, 0)])
        self.declare_partials('matrix_coeffs', ['G_W', 'g'],
                              rows=self._mass, cols=np.tile(ar, 5))
        self.declare_partials('matrix_coeffs', ['V', 'Theta_bar'],
                              rows=np.concatenate([flat(0, 2, 1),
                                                   flat(1, 2, 1)]),
                              cols=np.tile(ar, 2))

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        mat = np.zeros((nn, 3, 3, 3), dtype=inputs._get_data().dtype)
        mass = inputs['G_W'] / inputs['g']
        momentum = mass * inputs['V']

        for (row, col, power), name in ENTRIES.items():
            mat[:, row, col, power] = inputs[name]

        mat[:, 0, 0, 2] = -mass
        mat[:, 1, 1, 2] -= mass
        mat[:, 0, 2, 1] = inputs['dX_dq'] - momentum * inputs['Theta_bar']
        mat[:, 0, 2, 0] = -inputs['G_W']
        mat[:, 1, 2, 1] = inputs['dZ_dq'] + momentum
        mat[:, 2, 2, 2] = -inputs['I_yy']

        outputs['matrix_coeffs'] = mat

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        G_W, g, V = inputs['G_W'], inputs['g'], inputs['V']
        theta = inputs['Theta_bar']
        mass = G_W / g
        zero, one = np.zeros(nn), np.ones(nn)

        # order: (0,0,2) -m, (1,1,2) -m, (0,2,1) -mV theta, (1,2,1) +mV,
        #        (0,2,0) -G.W.
        J['matrix_coeffs', 'G_W'] = np.concatenate(
            [-one / g, -one / g, -V * theta / g, V / g, -one])
        J['matrix_coeffs', 'g'] = np.concatenate(
            [mass / g, mass / g, mass * V * theta / g, -mass * V / g, zero])
        J['matrix_coeffs', 'V'] = np.concatenate([-mass * theta, mass])
        J['matrix_coeffs', 'Theta_bar'] = np.concatenate([-mass * V, zero])
