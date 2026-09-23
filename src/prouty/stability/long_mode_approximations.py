"""Modal approximations to the longitudinal motion in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", pp. 623-626. The phugoid
reduction is p. 623 and its one-degree-of-freedom limit p. 624; the
short-period reduction is p. 625.
"""

import numpy as np
import openmdao.api as om


class _TwoByTwoMatrixComp(om.ExplicitComponent):
    """Shared plumbing for the two reductions: a 2x2 polynomial matrix."""

    #: (row, column, power of s) -> input name, for single-derivative entries.
    ENTRIES = {}
    #: Inputs and their units.
    DERIVATIVE_UNITS = {}

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def _setup_common(self, extra_inputs):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        shape = (nn, 2, 2, 3)

        for name, units in self.DERIVATIVE_UNITS.items():
            self.add_input(name, val=np.zeros(nn), units=units)
        for name, (value, units) in extra_inputs.items():
            self.add_input(name, val=np.full(nn, value), units=units)

        self.add_output('matrix_coeffs', shape=shape)

        def flat(row, col, power):
            return np.ravel_multi_index(
                (ar, np.full(nn, row), np.full(nn, col), np.full(nn, power)),
                shape)

        for (row, col, power), name in self.ENTRIES.items():
            self.declare_partials('matrix_coeffs', name,
                                  rows=flat(row, col, power), cols=ar, val=1.0)
        return ar, flat


class PhugoidMatrixComp(_TwoByTwoMatrixComp):
    """The phugoid reduction of p. 623: drop the Z equation.

    p. 623: "the observation that the phugoid involves little change in angle
    of attack allows its analysis to be made by eliminating the Z equation".
    What is left is speed and pitch, states ``x(s)`` and ``Theta(s)``::

        | -m s^2 + dX/dxdot s      (dX/dq - m V Theta_bar)s - G.W. |
        | dM/dxdot s               -I_yy s^2 + dM/dq s             |

    Feed it to ``PolyDeterminantComp(n=2, degree=2, n_zero_roots=1)``.

    The sign the book gets wrong
    ----------------------------
    p. 623 prints the resulting cubic with

        (1/(G.W./g))[(dX/dxdot)(dM/dq) **+** (dX/dq - (G.W./g) V Theta)(dM/dxdot)] s

    and expanding the determinant above gives a **minus** on the second
    product. The book's own numbers settle it: p. 624 tabulates the
    "full 2 degrees of freedom" phugoid for the 54 ft² stabilizer at
    ``omega = .342``, ``P = 18.4 s``, and the minus gives ``.3438`` and
    ``18.4 s`` where the printed plus gives ``.3487`` and ``18.0 s``.

    The same sign propagates into p. 624's one-degree-of-freedom reduction.
    Entry C9-19 of ``docs/validation_stability.md``. This component builds the
    matrix, so the determinant produces the corrected form by construction.

    The third approximation is already in the repository
    ----------------------------------------------------
    p. 624 goes one step further and drops ``I_yy``, on the same argument
    Hohenemser used in hover, leaving

        omega_nat = sqrt(-g (dM/dxdot)/(dM/dq))

    and says it "is the same equation as that derived for hover".
    ``HohenemserPeriodComp`` already computes it; there is no new component
    for that row.

    Options
    -------
    num_nodes : int

    Example helicopter with a 54 ft² stabilizer
    -------------------------------------------
    ================================= ========== ========
    p. 624                            model      book
    ================================= ========== ========
    full 3 degrees of freedom         .3653      .365
    full 2 degrees of freedom         .3438      .342
    approximate 2 degrees of freedom  .3556      .356
    ================================= ========== ========
    """

    ENTRIES = {(0, 0, 1): 'dX_dxdot', (1, 0, 1): 'dM_dxdot',
               (1, 1, 1): 'dM_dq'}
    DERIVATIVE_UNITS = {'dX_dxdot': 'lbf*s/ft', 'dX_dq': 'lbf*s/rad',
                        'dM_dxdot': 'lbf*s', 'dM_dq': 'lbf*ft*s/rad'}

    def setup(self):
        ar, flat = self._setup_common({
            'G_W': (20000.0, 'lbf'), 'I_yy': (40000.0, 'slug*ft**2'),
            'g': (32.2, 'ft/s**2'), 'V': (194.1, 'ft/s'),
            'Theta_bar': (-0.016507, 'rad')})

        self.declare_partials('matrix_coeffs', 'dX_dq', rows=flat(0, 1, 1),
                              cols=ar, val=1.0)
        self.declare_partials('matrix_coeffs', 'I_yy', rows=flat(1, 1, 2),
                              cols=ar, val=-1.0)
        self.declare_partials('matrix_coeffs', ['G_W', 'g'],
                              rows=np.concatenate([flat(0, 0, 2),
                                                   flat(0, 1, 1),
                                                   flat(0, 1, 0)]),
                              cols=np.tile(ar, 3))
        self.declare_partials('matrix_coeffs', ['V', 'Theta_bar'],
                              rows=flat(0, 1, 1), cols=ar)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        mat = np.zeros((nn, 2, 2, 3), dtype=inputs._get_data().dtype)
        mass = inputs['G_W'] / inputs['g']

        for (row, col, power), name in self.ENTRIES.items():
            mat[:, row, col, power] = inputs[name]

        mat[:, 0, 0, 2] = -mass
        mat[:, 0, 1, 1] = (inputs['dX_dq']
                           - mass * inputs['V'] * inputs['Theta_bar'])
        mat[:, 0, 1, 0] = -inputs['G_W']
        mat[:, 1, 1, 2] = -inputs['I_yy']

        outputs['matrix_coeffs'] = mat

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        G_W, g, V = inputs['G_W'], inputs['g'], inputs['V']
        theta, mass = inputs['Theta_bar'], inputs['G_W'] / inputs['g']
        one = np.ones(nn)

        J['matrix_coeffs', 'G_W'] = np.concatenate(
            [-one / g, -V * theta / g, -one])
        J['matrix_coeffs', 'g'] = np.concatenate(
            [mass / g, mass * V * theta / g, np.zeros(nn)])
        J['matrix_coeffs', 'V'] = -mass * theta
        J['matrix_coeffs', 'Theta_bar'] = -mass * V


class ShortPeriodMatrixComp(_TwoByTwoMatrixComp):
    """The short-period reduction of p. 625: drop the X equation.

    p. 625: "the time associated with this mode is so short that it can be
    assumed that no speed change occurs while it is being excited". What is
    left is heave and pitch, states ``z(s)`` and ``Theta(s)``::

        | (dZ/dzddot - m)s^2 + dZ/dzdot s     (dZ/dq + m V)s        |
        | dM/dzddot s^2 + dM/dzdot s          -I_yy s^2 + dM/dq s   |

    Feed it to ``PolyDeterminantComp(n=2, degree=2, n_zero_roots=2)``.

    Unlike the phugoid reduction, the quadratic p. 625 prints is exactly what
    this determinant gives -- expanded term by term, every sign matches.

    It is also the better approximation. p. 625 says its roots "are very
    similar to the corresponding roots of the three-degree-of-freedom system",
    and for the example helicopter they are ``-2.549`` and ``+1.037`` against
    the full quartic's ``-2.564`` and ``+.9867``. The phugoid reduction on the
    same aircraft returns a complex pair where the full system has two real
    roots, which is the "sacrificed reasonableness for the phugoid damping"
    of p. 624.

    Options
    -------
    num_nodes : int
    """

    ENTRIES = {(0, 0, 1): 'dZ_dzdot', (1, 0, 1): 'dM_dzdot',
               (1, 0, 2): 'dM_dzddot', (1, 1, 1): 'dM_dq'}
    DERIVATIVE_UNITS = {'dZ_dzdot': 'lbf*s/ft', 'dZ_dzddot': 'lbf*s**2/ft',
                        'dZ_dq': 'lbf*s/rad', 'dM_dzdot': 'lbf*s',
                        'dM_dzddot': 'lbf*s**2', 'dM_dq': 'lbf*ft*s/rad'}

    def setup(self):
        ar, flat = self._setup_common({
            'G_W': (20000.0, 'lbf'), 'I_yy': (40000.0, 'slug*ft**2'),
            'g': (32.2, 'ft/s**2'), 'V': (194.1, 'ft/s')})

        self.declare_partials('matrix_coeffs', 'dZ_dzddot',
                              rows=flat(0, 0, 2), cols=ar, val=1.0)
        self.declare_partials('matrix_coeffs', 'dZ_dq', rows=flat(0, 1, 1),
                              cols=ar, val=1.0)
        self.declare_partials('matrix_coeffs', 'I_yy', rows=flat(1, 1, 2),
                              cols=ar, val=-1.0)
        self.declare_partials('matrix_coeffs', ['G_W', 'g', 'V'],
                              rows=np.concatenate([flat(0, 0, 2),
                                                   flat(0, 1, 1)]),
                              cols=np.tile(ar, 2))

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        mat = np.zeros((nn, 2, 2, 3), dtype=inputs._get_data().dtype)
        mass = inputs['G_W'] / inputs['g']

        for (row, col, power), name in self.ENTRIES.items():
            mat[:, row, col, power] = inputs[name]

        mat[:, 0, 0, 2] = inputs['dZ_dzddot'] - mass
        mat[:, 0, 1, 1] = inputs['dZ_dq'] + mass * inputs['V']
        mat[:, 1, 1, 2] = -inputs['I_yy']

        outputs['matrix_coeffs'] = mat

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        G_W, g, V = inputs['G_W'], inputs['g'], inputs['V']
        mass = G_W / g
        one = np.ones(nn)

        J['matrix_coeffs', 'G_W'] = np.concatenate([-one / g, V / g])
        J['matrix_coeffs', 'g'] = np.concatenate([mass / g, -mass * V / g])
        J['matrix_coeffs', 'V'] = np.concatenate([np.zeros(nn), mass * one])
