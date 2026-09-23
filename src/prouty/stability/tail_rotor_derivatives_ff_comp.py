"""Tail rotor dimensional stability derivatives in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.9, pp. 582-583. Its inputs
are the tail rotor column of Table 9.5 (p. 574) and Table 9.7 (p. 578).
"""

from prouty.stability.monomial_rows_comp import MonomialRowsComp, ref, term

#: rho A_b (Omega R)^2 for the tail rotor.
Q = ('rho', 'A_b', 'Omega_R', 'Omega_R')

SCALAR_INPUTS = ('A_b', 'h_T', 'l_T')

INPUT_UNITS = {
    'rho': 'slug/ft**3', 'A_b': 'ft**2', 'Omega_R': 'ft/s',
    'h_T': 'ft', 'l_T': 'ft',
    'd_mu_d_xdot': 's/ft', 'd_lambda_d_ydot': 's/ft',
}


def build_rows():
    """Table 9.9, pp. 582-583. Thirteen of the fifteen rows are references."""
    return {
        # ---- Y, p. 582 -------------------------------------------------
        'dY_dxdot': [term(1.0, Q + ('dCT_sigma_dmu', 'd_mu_d_xdot'))],
        'dY_dydot': [term(1.0, Q + ('dCT_sigma_dlambda', 'd_lambda_d_ydot'))],
        'dY_dp': [ref(1.0, 'dY_dydot', ('h_T',))],
        'dY_dr': [ref(-1.0, 'dY_dydot', ('l_T',))],
        'dY_dtheta0': [term(1.0, Q + ('dCT_sigma_dtheta0',))],

        # ---- R, p. 583: the side force carried on h_T ------------------
        'dR_dxdot': [ref(1.0, 'dY_dxdot', ('h_T',))],
        'dR_dydot': [ref(1.0, 'dY_dydot', ('h_T',))],
        'dR_dp': [ref(1.0, 'dY_dp', ('h_T',))],
        'dR_dr': [ref(1.0, 'dY_dr', ('h_T',))],
        'dR_dtheta0': [ref(1.0, 'dY_dtheta0', ('h_T',))],

        # ---- N, p. 583: the same carried on -l_T -----------------------
        'dN_dxdot': [ref(-1.0, 'dY_dxdot', ('l_T',))],
        'dN_dydot': [ref(-1.0, 'dY_dydot', ('l_T',))],
        'dN_dp': [ref(-1.0, 'dY_dp', ('l_T',))],
        'dN_dr': [ref(-1.0, 'dY_dr', ('l_T',))],
        'dN_dtheta0': [ref(-1.0, 'dY_dtheta0', ('l_T',))],
    }


class TailRotorDerivativesFFComp(MonomialRowsComp):
    """Table 9.9: the tail rotor's 15 dimensional derivatives at 115 knots.

    Structurally the simplest table in the chapter. Two rows are aerodynamic::

        dY/dxdot = rho A_b (Omega R)^2 (dCT/sigma/dmu)(dmu/dxdot)
        dY/dydot = rho A_b (Omega R)^2 (dCT/sigma/dlambda')(dlambda'/dydot)

    a third is the collective row, and the remaining twelve are those three
    carried to the c.g. on ``h_T`` for rolling moment and ``-l_T`` for yawing
    moment. Nothing else happens.

    What changed from hover
    -----------------------
    Table 9.3 and Table 9.9 both have fifteen rows, but not the same fifteen.
    Forward flight **adds** the three ``xdot`` rows -- a tail rotor in forward
    flight sees its own advance ratio change -- and **drops all three M rows**,
    the ones that came from tail rotor torque reacting about the shaft axis.

    Dropping them sidesteps the contradiction of entry C9-9, where Table 9.3's
    ``dM/dydot`` and ``dM/dtheta0`` carried opposite signs for the same
    physics. Chapter 9 never resolves it; it simply stops printing those rows.

    It settles C9-8 instead
    -----------------------
    Table 9.3 prints ``(dR/dydot)_T = (dY/dydot)_T h_T`` and then a value of
    **+78** where that expression gives -78, and the error propagates into
    Table 9.4's total.

    Table 9.9 prints the same equation for the same derivative and a value of
    **-147**, which is ``-24.5 x 6`` exactly. So the forward-flight table gets
    the sign right where the hover table does not, using the same ``h_T = +6``.
    That is independent confirmation that the hover +78 is a misprint rather
    than a convention this implementation has misread.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    All fifteen printed rows reproduce. Nine are exact, and the rest are
    limited by Table 9.7's ``dlambda'/dydot`` and by single-digit rounding on
    ``dY/dxdot``.

    One internal rounding is worth naming. ``dR/dr = (dY/dr) h_T = 5,442`` and
    ``dN/dp = -(dY/dp) l_T = 5,439`` are the same product,
    ``24.5 x 6 x 37``, reached two ways. The book prints both, differing by
    0.06 % because ``dY/dr`` was rounded to 907 before being multiplied.
    """

    scalar_inputs = SCALAR_INPUTS
    input_units = INPUT_UNITS
    input_defaults = {'rho': 0.002378, 'A_b': 19.397, 'Omega_R': 650.0,
                      'h_T': 6.0, 'l_T': 37.0}

    def rows(self):
        return build_rows()
