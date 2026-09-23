"""The two-degree-of-freedom longitudinal hover equations, as a matrix.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", p. 598, with the same system drawn
as a block diagram in Figure 9.8 and as a mechanical analog in Figure 9.9,
both p. 599.
"""

import numpy as np
import openmdao.api as om

#: (row, column, power of s) -> input name.
ENTRIES = {
    (0, 0, 0): 'dX_dxdot',
    (0, 1, 1): 'dX_dq',
    (1, 0, 0): 'dM_dxdot',
    (1, 1, 1): 'dM_dq',
}

DERIVATIVE_UNITS = {
    'dX_dxdot': 'lbf*s/ft', 'dX_dq': 'lbf*s/rad',
    'dM_dxdot': 'lbf*s', 'dM_dq': 'lbf*ft*s/rad',
}


class HoverLongTwoDofMatrixComp(om.ExplicitComponent):
    """The 2x2 system of p. 598: the hovering helicopter held vertically.

    Dropping the Z-force equation leaves fore-and-aft translation and pitch::

        | dX/dxdot - m s      (dX/dq)s - G.W.      |
        | dM/dxdot            (dM/dq)s - I_yy s^2  |

    with ``m = G.W./g`` and the state ``[xdot(s), Theta(s)]``. Feed it to
    ``PolyDeterminantComp(n=2, degree=2, degree_out=3)``; only the ``I_yy``
    entry is quadratic, so the determinant is a cubic where the entrywise
    bound would say quartic.

    Why it is worth having alongside the 3x3
    ----------------------------------------
    p. 600 makes the point: the roots come out at -.87 and .075 +/- .355i
    against the full set's -.89, -.28 and .076 +/- .360i. The only root lost is
    the plunge convergence, and the oscillation -- the mode that actually
    matters to a hovering pilot -- moves by under 2 %. It is also the form
    Routh's discriminant is applied to on p. 602, since R.D.(3) needs a cubic.

    The signs, checked three ways
    -----------------------------
    p. 598 writes ``-(G.W./g) x_ddot + ... - G.W.Theta`` and
    ``... - I_yy q_dot + ...``. Figure 9.9 prints the same two equations with
    the inertia moved to the right-hand side, and Figure 9.8 feeds ``-G.W.``
    from ``Theta`` into the X-force summer and ``1/I_yy`` into ``q_dot``. All
    three agree, which is worth noting because the *three*-degree-of-freedom
    Z-force equation on p. 596 does not (entry C9-11).

    The missing s term
    ------------------
    The printed cubic has no ``s`` term at all::

        s^3 - [(1/m)(dX/dxdot) + (1/I_yy)(dM/dq)] s^2 + (g/I_yy)(dM/dxdot) = 0

    The determinant does produce one, ``(dX/dxdot)(dM/dq) - (dX/dq)(dM/dxdot)``,
    and p. 597 shows it cancels identically for a single rotor. Nothing here
    forces it to zero. With the two-significant-figure Table 9.4 values it
    comes out at -3.4e-5 against an ``s^2`` coefficient of .72, which is the
    rounding of those values and not a modelling choice.

    Options
    -------
    num_nodes : int

    Example helicopter
    ------------------
    From Table 9.4 with ``G.W. = 20,000 lb`` and ``I_yy = 40,000 slug ft^2``:
    ``s^3 + .7245 s^2 + .1151 = 0`` against the printed
    ``s^3 + .724 s^2 + .115 = 0``, with roots -.872 and .0736 +/- .3556i
    against -.87 and .075 +/- .355i.
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
        self.add_input('I_yy', val=np.full(nn, 40000.0), units='slug*ft**2')
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
        self.declare_partials('matrix_coeffs', 'I_yy', rows=flat(1, 1, 2),
                              cols=ar, val=-1.0)

    def compute(self, inputs, outputs):
        nn = self.options['num_nodes']
        mat = np.zeros((nn, 2, 2, 3), dtype=inputs._get_data().dtype)

        for (row, col, power), name in ENTRIES.items():
            mat[:, row, col, power] = inputs[name]

        mat[:, 0, 0, 1] = -inputs['G_W'] / inputs['g']
        mat[:, 0, 1, 0] = -inputs['G_W']
        mat[:, 1, 1, 2] = -inputs['I_yy']

        outputs['matrix_coeffs'] = mat

    def compute_partials(self, inputs, J):
        G_W, g = inputs['G_W'], inputs['g']
        nn = self.options['num_nodes']

        J['matrix_coeffs', 'G_W'] = np.concatenate([-1.0 / g, -np.ones(nn)])
        J['matrix_coeffs', 'g'] = np.concatenate([G_W / g ** 2, np.zeros(nn)])
