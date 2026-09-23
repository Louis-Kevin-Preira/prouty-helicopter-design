"""The longitudinal equations of motion in hover, as a polynomial matrix.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 596-598. The three equations
are on p. 596, the determinant on p. 597, and its roots on p. 598.
"""

import numpy as np
import openmdao.api as om

#: (row, column, power of s) -> (input name, sign). Anything not listed is
#: structural and handled in compute directly.
ENTRIES = {
    (0, 0, 0): ('dX_dxdot', 1.0),
    (0, 1, 0): ('dX_dzdot', 1.0),
    (0, 2, 1): ('dX_dq', 1.0),
    (1, 0, 0): ('dZ_dxdot', 1.0),
    (1, 1, 0): ('dZ_dzdot', 1.0),
    (1, 1, 1): ('dZ_dzddot', 1.0),
    (1, 2, 1): ('dZ_dq', 1.0),
    (2, 0, 0): ('dM_dxdot', 1.0),
    (2, 1, 0): ('dM_dzdot', 1.0),
    (2, 1, 1): ('dM_dzddot', 1.0),
    (2, 2, 1): ('dM_dq', 1.0),
}

DERIVATIVE_UNITS = {
    'dX_dxdot': 'lbf*s/ft', 'dX_dzdot': 'lbf*s/ft', 'dX_dq': 'lbf*s/rad',
    'dZ_dxdot': 'lbf*s/ft', 'dZ_dzdot': 'lbf*s/ft',
    'dZ_dzddot': 'lbf*s**2/ft', 'dZ_dq': 'lbf*s/rad',
    'dM_dxdot': 'lbf*s', 'dM_dzdot': 'lbf*s', 'dM_dzddot': 'lbf*s**2',
    'dM_dq': 'lbf*ft*s/rad',
}

#: Derivatives p. 597 states are zero in hover.
ZERO_IN_HOVER = ('dX_dzdot', 'dZ_dxdot', 'dZ_dzddot', 'dZ_dq', 'dM_dzddot')


class HoverLongitudinalMatrixComp(om.ExplicitComponent):
    """The 3x3 determinant of p. 597, in the form ``PolyDeterminantComp`` eats.

    The state vector is ``[xdot(s), zdot(s), Theta(s)]`` -- two velocities and
    an angle, not three displacements, which is why the determinant is already
    a quartic and carries no ``s**k`` rigid-body factor to divide out::

        | dX/dxdot - m s        dX/dzdot                  (dX/dq)s - G.W. |
        | dZ/dxdot              dZ/dzdot + (dZ/dzddot-m)s (dZ/dq)s        |
        | dM/dxdot              dM/dzdot + (dM/dzddot)s   (dM/dq)s - Iyy s^2 |

    with ``m = G.W./g``. Feed the output to ``PolyDeterminantComp(n=3,
    degree=2, degree_out=4)``; the degree bound would say six, but only the
    ``Iyy s^2`` entry is quadratic, so the two top coefficients vanish
    identically and ``degree_out`` keeps ``normalize`` from dividing by one of
    them.

    What is zero in hover
    ---------------------
    p. 597 lists five derivatives as zero: ``dX/dzdot``, ``dZ/dxdot``,
    ``dZ/dzddot``, ``dZ/dq`` and ``dM/dzddot``. They are still inputs, so the
    same component serves a case where they are not, and they default to zero.

    ``dM/dzdot`` is **not** among them -- Table 9.4 gives it as 91 -- but it
    drops out of the determinant anyway: with ``dZ/dxdot`` and ``dZ/dq`` zero
    the middle row is ``[0, ., 0]``, and expanding along it removes the whole
    ``zdot`` column from every other term. That is why the characteristic
    equation of p. 597 contains only ``dX/dxdot``, ``dZ/dzdot``, ``dM/dq`` and
    ``dM/dxdot``.

    The cross terms that cancel
    ---------------------------
    p. 597 notes that ``(dM/dq)(dX/dxdot) - (dM/dxdot)(dX/dq)`` vanishes for a
    single rotor, which is why the printed quartic has no ``s`` term beyond
    ``(g/Iyy)(dM/dxdot)``. It follows from Table 9.2: both ``dX`` derivatives
    carry the same ``dCH/sigma/da1s`` factor and both ``dM`` derivatives the
    same ``dM/da1s`` and ``-h_M``, so the two products are identical. Nothing
    here enforces it; it simply comes out of the arithmetic, which makes the
    printed quartic an independent check on Table 9.2.

    The sign of the mass term
    -------------------------
    p. 596 prints the Z-force equation with ``+(G.W./g) z_ddot`` while the X
    and M equations both carry their inertia with a minus, and it omits
    ``dZ/dzddot`` entirely. The matrix on p. 597 has ``[dZ/dzddot - G.W./g]s``.
    The matrix is the right one: the plus would put the plunge root at
    ``-dZ/dzdot/m = +0.293``, a divergence, where p. 598 lists ``-.28``.
    Entry C9-11 of ``docs/validation_stability.md``.

    Options
    -------
    num_nodes : int

    Example helicopter
    ------------------
    Fed Table 9.4 with ``G.W. = 20,000 lb`` and ``I_yy = 40,000 slug ft^2``,
    the determinant is ``s^4 + 1.0176 s^3 + .2123 s^2 + .1151 s + .0337``
    against the printed ``s^4 + 1.02 s^3 + .215 s^2 + .12 s + .034``.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        shape = (nn, 3, 3, 3)

        for name, units in DERIVATIVE_UNITS.items():
            default = 0.0 if name in ZERO_IN_HOVER else 1.0
            self.add_input(name, val=np.full(nn, default), units=units)
        self.add_input('G_W', val=np.full(nn, 20000.0), units='lbf')
        self.add_input('I_yy', val=np.full(nn, 40000.0), units='slug*ft**2')
        self.add_input('g', val=np.full(nn, 32.2), units='ft/s**2')

        self.add_output('matrix_coeffs', shape=shape)

        def flat(row, col, power):
            return np.ravel_multi_index(
                (ar, np.full(nn, row), np.full(nn, col), np.full(nn, power)),
                shape)

        for (row, col, power), (name, sign) in ENTRIES.items():
            self.declare_partials('matrix_coeffs', name,
                                  rows=flat(row, col, power), cols=ar,
                                  val=sign)

        # -m s in rows 0 and 1, -G.W. in the X row, -I_yy s^2 in the M row
        self._mass_rows = np.concatenate([flat(0, 0, 1), flat(1, 1, 1)])
        self.declare_partials('matrix_coeffs', ['G_W', 'g'],
                              rows=np.concatenate([self._mass_rows,
                                                   flat(0, 2, 0)]),
                              cols=np.tile(ar, 3))
        self.declare_partials('matrix_coeffs', 'I_yy', rows=flat(2, 2, 2),
                              cols=ar, val=-1.0)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        mat = np.zeros((nn, 3, 3, 3), dtype=inputs._get_data().dtype)
        mass = inputs['G_W'] / inputs['g']

        for (row, col, power), (name, sign) in ENTRIES.items():
            mat[:, row, col, power] = sign * inputs[name]

        mat[:, 0, 0, 1] -= mass
        mat[:, 1, 1, 1] -= mass
        mat[:, 0, 2, 0] = -inputs['G_W']
        mat[:, 2, 2, 2] = -inputs['I_yy']

        outputs['matrix_coeffs'] = mat

    def compute_partials(self, inputs, J):
        G_W, g = inputs['G_W'], inputs['g']
        nn = self.options['num_nodes']

        # two -m s entries then the -G.W. entry
        J['matrix_coeffs', 'G_W'] = np.concatenate(
            [-1.0 / g, -1.0 / g, -np.ones(nn)])
        J['matrix_coeffs', 'g'] = np.concatenate(
            [G_W / g ** 2, G_W / g ** 2, np.zeros(nn)])
