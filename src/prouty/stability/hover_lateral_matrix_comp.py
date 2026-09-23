"""The lateral equations of motion in hover, as a polynomial matrix.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", p. 604. The page prints no
equations; it says the lateral mode "could have been treated in the same
manner by using the moment of inertia in roll instead of pitch". The signs of
the resulting matrix are taken from Table 9.19, pp. 614-615, where the lateral
equations *are* written out, with the forward speed set to zero.
"""

import numpy as np
import openmdao.api as om

#: (row, column, power of s) -> input name.
ENTRIES = {
    (0, 0, 0): 'dY_dydot',
    (0, 1, 1): 'dY_dp',
    (1, 0, 0): 'dR_dydot',
    (1, 1, 1): 'dR_dp',
}

DERIVATIVE_UNITS = {
    'dY_dydot': 'lbf*s/ft', 'dY_dp': 'lbf*s/rad',
    'dR_dydot': 'lbf*s', 'dR_dp': 'lbf*ft*s/rad',
}


class HoverLateralMatrixComp(om.ExplicitComponent):
    """Roll and sideways translation in hover, with heading held.

    p. 604 asks the reader to redo the longitudinal analysis with ``I_xx``.
    Two degrees of freedom, ``[ydot(s), Phi(s)]``::

        | dY/dydot - m s      (dY/dp)s + G.W.      |
        | dR/dydot            (dR/dp)s - I_xx s^2  |

    with ``m = G.W./g``. Yaw is not in it: p. 604 has the pilot holding heading
    on the pedals, and p. 605 treats the yaw axis separately as a
    single-degree-of-freedom mode. Feed the output to
    ``PolyDeterminantComp(n=2, degree=2, degree_out=3)``.

    The gravity term is +G.W., not -G.W.
    ------------------------------------
    This is the one sign the analogy does not hand you, and p. 604 does not
    print it. Transposing the longitudinal equations symbol for symbol would
    give ``-G.W.``; the correct value is ``+G.W.``.

    Table 9.19 (p. 615) settles it: the Y row of the Phi column reads
    ``(dY/dp + (G.W./g) V_bar Theta_bar)s + G.W.``, against ``(dX/dq - ...)s -
    G.W.`` in the X row of the Theta column. Table 9.20 confirms it in numbers
    on the facing page: ``-3964 s + 20000`` for Y against ``3927 s - 20000``
    for X, with ``G.W. = 20,000 lb``.

    It is also what the axes require. With ``x`` forward, ``y`` right and ``z``
    down, a nose-up ``Theta`` puts ``-W sin(Theta)`` along body ``x``, while a
    right-wing-down ``Phi`` puts ``+W sin(Phi)`` along body ``y``. The
    consequence is that the constant term of the lateral cubic is
    ``-(g/I_xx)(dR/dydot)`` where the longitudinal one is
    ``+(g/I_yy)(dM/dxdot)``.

    No cross-term cancellation here
    -------------------------------
    The longitudinal cubic loses its ``s`` term because ``dX/dxdot`` and
    ``dX/dq`` share a factor and so do ``dM/dxdot`` and ``dM/dq`` (p. 597).
    The lateral one keeps its ``s`` term: ``dY/dydot`` and ``dR/dydot`` carry
    tail rotor contributions, and those are proportional to nothing the roll
    derivatives contain. Restricted to the main rotor alone the cancellation
    reappears, which is a useful check that the difference is the tail rotor
    and not an error.

    Options
    -------
    num_nodes : int

    Example helicopter
    ------------------
    From Table 9.4 with ``G.W. = 20,000 lb`` and ``I_xx = 5,000 slug ft^2``
    (Table 9.20's ``-5000 s^2``), the cubic is
    ``s^3 + 5.854 s^2 + .1461 s + .4186`` with roots -5.84 and
    -.0064 +/- .2676i: a fast roll convergence and a slow, barely damped
    lateral oscillation of period 23.5 s.

    That result depends on ``dR/dydot``, which is the derivative entry C9-8 is
    about. With Table 9.4's printed -65 the oscillation is marginally stable;
    with the -222 the tail rotor equation actually gives, it is marginally
    unstable. See the validation notes.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']
        ar = np.arange(nn)
        shape = (nn, 2, 2, 3)

        for name, units in DERIVATIVE_UNITS.items():
            self.add_input(name, val=np.ones(nn), units=units)
        self.add_input('G_W', val=np.full(nn, 20000.0), units='lbf')
        self.add_input('I_xx', val=np.full(nn, 5000.0), units='slug*ft**2')
        self.add_input('g', val=np.full(nn, 32.2), units='ft/s**2')

        self.add_output('matrix_coeffs', shape=shape)

        def flat(row, col, power):
            return np.ravel_multi_index(
                (ar, np.full(nn, row), np.full(nn, col), np.full(nn, power)),
                shape)

        for (row, col, power), name in ENTRIES.items():
            self.declare_partials('matrix_coeffs', name,
                                  rows=flat(row, col, power), cols=ar, val=1.0)

        self.declare_partials('matrix_coeffs', ['G_W', 'g'],
                              rows=np.concatenate([flat(0, 0, 1),
                                                   flat(0, 1, 0)]),
                              cols=np.tile(ar, 2))
        self.declare_partials('matrix_coeffs', 'I_xx', rows=flat(1, 1, 2),
                              cols=ar, val=-1.0)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        mat = np.zeros((nn, 2, 2, 3), dtype=inputs._get_data().dtype)

        for (row, col, power), name in ENTRIES.items():
            mat[:, row, col, power] = inputs[name]

        mat[:, 0, 0, 1] = -inputs['G_W'] / inputs['g']
        mat[:, 0, 1, 0] = inputs['G_W']                  # +G.W., see docstring
        mat[:, 1, 1, 2] = -inputs['I_xx']

        outputs['matrix_coeffs'] = mat

    def compute_partials(self, inputs, J):
        G_W, g = inputs['G_W'], inputs['g']
        nn = self.options['num_nodes']

        J['matrix_coeffs', 'G_W'] = np.concatenate([-1.0 / g, np.ones(nn)])
        J['matrix_coeffs', 'g'] = np.concatenate([G_W / g ** 2, np.zeros(nn)])
