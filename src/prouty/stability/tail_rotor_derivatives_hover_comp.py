"""Tail rotor dimensional stability derivatives near hover.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.3, pp. 569-570. Totals
cross-checked against Table 9.4, pp. 571-573. The tail rotor torque sign
convention is Chapter 8, p. 487.
"""

from prouty.stability.monomial_rows_comp import MonomialRowsComp, term

#: rho A_b (Omega R)^2 for the tail rotor.
Q = ('rho', 'A_b', 'Omega_R', 'Omega_R')
#: The same with the tail rotor radius, for the torque-derived moments.
QR = Q + ('R',)

#: The side-force derivative every geometric row is built from,
#: rho A_b (Omega R)^2 (dCT/sigma/dlambda')(dlambda'/dydot).
Y_YDOT = Q + ('dCT_sigma_dlambda', 'd_lambda_d_ydot')
#: rho A_b (Omega R)^2 (dCT/sigma/dtheta0).
Y_THETA0 = Q + ('dCT_sigma_dtheta0',)

SCALAR_INPUTS = ('A_b', 'R', 'h_T', 'l_T')

#: Must match the units the Table 9.1 component declares on its outputs.
INPUT_UNITS = {
    'rho': 'slug/ft**3', 'A_b': 'ft**2', 'Omega_R': 'ft/s', 'R': 'ft',
    'h_T': 'ft', 'l_T': 'ft', 'd_lambda_d_ydot': 's/ft',
}


def build_rows(torque_sign):
    """Table 9.3, with ``torque_sign`` on the two torque-derived moment rows."""
    s = torque_sign
    return {
        # ---- Y, p. 569 -------------------------------------------------
        # rho A_b (Omega R)^2 (dCT/sigma/dlambda')(dlambda'/dydot)
        'dY_dydot': [term(1.0, Y_YDOT)],
        'dY_dp': [term(1.0, Y_YDOT + ('h_T',))],           # (dY/dydot) h_T
        'dY_dr': [term(-1.0, Y_YDOT + ('l_T',))],          # -(dY/dydot) l_T
        # rho A_b (Omega R)^2 (dCT/sigma/dtheta0)
        'dY_dtheta0': [term(1.0, Y_THETA0)],

        # ---- R, pp. 569-570 --------------------------------------------
        'dR_dydot': [term(1.0, Y_YDOT + ('h_T',))],        # (dY/dydot) h_T
        'dR_dp': [term(1.0, Y_YDOT + ('h_T', 'h_T'))],     # (dY/dp) h_T
        'dR_dr': [term(-1.0, Y_YDOT + ('l_T', 'h_T'))],    # (dY/dr) h_T
        'dR_dtheta0': [term(1.0, Y_THETA0 + ('h_T',))],    # (dY/dtheta0) h_T

        # ---- M, p. 570 -------------------------------------------------
        # -s rho A_b (Omega R)^2 R (dCQ/sigma/dlambda')(dlambda'/dydot)
        'dM_dydot': [term(-s, QR + ('dCQ_sigma_dlambda', 'd_lambda_d_ydot'))],
        # -(dM/dydot) l_T
        'dM_dr': [term(s, QR + ('dCQ_sigma_dlambda', 'd_lambda_d_ydot',
                                'l_T'))],
        'dM_dtheta0': [term(-s, QR + ('dCQ_sigma_dtheta0',))],

        # ---- N, p. 570 -------------------------------------------------
        'dN_dydot': [term(-1.0, Y_YDOT + ('l_T',))],       # -(dY/dydot) l_T
        'dN_dp': [term(-1.0, Y_YDOT + ('h_T', 'l_T'))],    # -(dY/dp) l_T
        'dN_dr': [term(1.0, Y_YDOT + ('l_T', 'l_T'))],     # -(dY/dr) l_T
        'dN_dtheta0': [term(-1.0, Y_THETA0 + ('l_T',))],   # -(dY/dtheta0) l_T
    }


class TailRotorDerivativesHoverComp(MonomialRowsComp):
    """Table 9.3: the tail rotor's 15 dimensional derivatives in hover.

    Far simpler than the main rotor's Table 9.2. The tail rotor contributes no
    X or Z derivative worth keeping in hover, and every geometric row is one
    of two quantities carried to the CG on the arms ``h_T`` and ``l_T``::

        dY/dydot   = rho A_b (Omega R)^2 (dCT/sigma/dlambda')(dlambda'/dydot)
        dY/dtheta0 = rho A_b (Omega R)^2 (dCT/sigma/dtheta0)

    ``A_b``, ``Omega_R``, ``R`` and the two chart derivatives are the *tail
    rotor's*, so this component is fed by a tail rotor instance of
    ``BasicRotorDerivativesHoverComp``, in particular its ``d_lambda_d_ydot``,
    which is ``-1/(Omega R)_T`` (Table 9.1, p. 564).

    Options
    -------
    num_nodes : int
    blade_closest : {'up', 'down'}
        Which way the tail rotor blade nearest the main rotor travels, the same
        option and the same ``s = +1`` / ``s = -1`` as
        ``TailRotorForcesComp``. p. 487 gives ``M_T = -s Q_T``, so ``s`` sets
        the sign of the two rows that come from tail rotor torque, ``dM/dydot``
        and ``dM/dtheta0``, and through the first of them ``dM/dr``. It touches
        nothing else: Table 9.3 has no Z rows for it to reach.

        Default ``'up'``, which reproduces the printed ``dM/dydot = -7`` and
        ``dM/dr = 274``. See C9-9 below for what it does to ``dM/dtheta0``.

    Geometry the table implies
    --------------------------
    ``h_T = 6 ft`` and ``l_T = 37 ft``, each fixed exactly by three separate
    rows: ``dR/dp = -468``, ``dR/dr = 2,886`` and ``dR/dtheta0 = 58,476`` for
    the first, ``dN/dr = -17,797``, ``dN/dtheta0 = -360,602`` and
    ``dM/dr = 274`` for the second. ``l_T = 37`` is the same value Chapter 8
    uses. ``rho A_b (Omega R)^2 = 19,492`` and ``R = 6.5 ft`` come from
    ``dY/dtheta0 = 9,746`` and ``dM/dtheta0 = 11,276``.

    Two rows the book prints against its own equations
    --------------------------------------------------
    ``dR/dydot`` is printed as ``(dY/dydot)_T h_T`` with a value of **+78**.
    With ``dY/dydot = -13`` two rows above and ``h_T = +6``, that expression is
    -78. The three sibling rows built the same way are all exact, so the value
    is a sign misprint, and Table 9.4 carries it into a total of -65 where -221
    belongs. Entry C9-8 of ``docs/validation_stability.md``.

    ``dM/dtheta0`` is printed as ``+11,276``, and ``dM/dydot`` as ``-7``. Both
    come from tail rotor torque and cannot have opposite signs. The printed
    equations omit the ``-s`` of p. 487; applying it consistently with
    ``s = +1`` reproduces ``dM/dydot = -7`` and ``dM/dr = 274`` and turns
    ``dM/dtheta0`` into -11,276. Entry C9-9.

    Example helicopter
    ------------------
    Thirteen of the fifteen rows reproduce the printed value. The two
    exceptions are the ones above. Rows downstream of ``dY/dydot`` sit 1.5 %
    high throughout, because Prouty rounds it from -13.19 to the -13 he prints
    before carrying it onto the arms.
    """

    scalar_inputs = SCALAR_INPUTS
    input_units = INPUT_UNITS

    def initialize(self):
        super().initialize()
        self.options.declare('blade_closest', values=('up', 'down'),
                             default='up')

    def rows(self):
        return build_rows(1.0 if self.options['blade_closest'] == 'up'
                          else -1.0)
