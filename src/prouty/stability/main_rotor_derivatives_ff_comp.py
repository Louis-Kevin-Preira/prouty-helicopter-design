"""Main rotor dimensional stability derivatives in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.8, pp. 578-582. Its inputs
are the chart partials of Table 9.5 (p. 574) and the basic derivatives of
Table 9.6 (pp. 576-577). Totals cross-checked against Tables 9.19 and 9.20,
pp. 614-615.
"""

from prouty.stability.monomial_rows_comp import MonomialRowsComp, ref, term

#: rho A_b (Omega R)^2, and the same with the radius for the torque rows.
Q = ('rho', 'A_b', 'Omega_R', 'Omega_R')
QR = Q + ('R',)

SCALAR_INPUTS = ('A_b', 'R', 'h_M', 'l_M', 'y_M', 'i_M')

INPUT_UNITS = {
    'rho': 'slug/ft**3', 'A_b': 'ft**2', 'Omega_R': 'ft/s', 'Omega': 'rad/s',
    'R': 'ft', 'h_M': 'ft', 'l_M': 'ft', 'y_M': 'ft',
    'd_mu_d_xdot': 's/ft', 'd_lambda_d_xdot': 's/ft',
    'd_lambda_d_zdot': 's/ft', 'd_beta_d_ydot': 'rad*s/ft',
    'd_a1s_dq': 's', 'd_a1s_dp': 's', 'd_b1s_dq': 's', 'd_b1s_dp': 's',
    'dM_da1s': 'lbf*ft/rad', 'dR_db1s': 'lbf*ft/rad',
    'a1s_bar': 'rad', 'b1s_bar': 'rad', 'i_M': 'rad',
    'A_1': 'rad', 'B_1': 'rad',
}


def build_rows():
    """Table 9.8, pp. 578-582, in the nested form the book prints."""
    return {
        # ---- X, pp. 578-579 --------------------------------------------
        # -rho A_b (Omega R)^2 {[dCH/dmu + (dCH/da1s)(da1s/dmu)
        #   + (a1s_bar + i_M)(dCT/dmu)] dmu/dxdot + [same in lambda'] dlambda'/dxdot}
        'dX_dxdot': [
            term(-1.0, Q + ('dCH_sigma_dmu', 'd_mu_d_xdot')),
            term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_d_mu', 'd_mu_d_xdot')),
            term(-1.0, Q + ('a1s_bar', 'dCT_sigma_dmu', 'd_mu_d_xdot')),
            term(-1.0, Q + ('i_M', 'dCT_sigma_dmu', 'd_mu_d_xdot')),
            term(-1.0, Q + ('dCH_sigma_dlambda', 'd_lambda_d_xdot')),
            term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_d_lambda',
                            'd_lambda_d_xdot')),
            term(-1.0, Q + ('a1s_bar', 'dCT_sigma_dlambda', 'd_lambda_d_xdot')),
            term(-1.0, Q + ('i_M', 'dCT_sigma_dlambda', 'd_lambda_d_xdot')),
        ],
        # +rho A_b (Omega R)^2 (C_T_bar/sigma)(A_1 - b1s_bar) dbeta/dydot
        'dX_dydot': [
            term(1.0, Q + ('CT_sigma_bar', 'A_1', 'd_beta_d_ydot')),
            term(-1.0, Q + ('CT_sigma_bar', 'b1s_bar', 'd_beta_d_ydot')),
        ],
        # -rho A_b (Omega R)^2 [dCH/dlambda' + (C_T_bar/sigma)(da1s/dlambda')
        #   + (a1s_bar + i_M)(dCT/dlambda')] dlambda'/dzdot   -- see C9-16
        'dX_dzdot': [
            term(-1.0, Q + ('dCH_sigma_dlambda', 'd_lambda_d_zdot')),
            term(-1.0, Q + ('CT_sigma_bar', 'd_a1s_d_lambda',
                            'd_lambda_d_zdot')),
            term(-1.0, Q + ('a1s_bar', 'dCT_sigma_dlambda', 'd_lambda_d_zdot')),
            term(-1.0, Q + ('i_M', 'dCT_sigma_dlambda', 'd_lambda_d_zdot')),
        ],
        # -rho A_b (Omega R)^2 (dCH/da1s)(da1s/dq) - (dX/dxdot) h_M
        'dX_dq': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dq')),
                  ref(-1.0, 'dX_dxdot', ('h_M',))],
        'dX_dp': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dp')),
                  ref(-1.0, 'dX_dydot', ('h_M',))],
        'dX_dtheta0': [
            term(-1.0, Q + ('dCH_sigma_dtheta0',)),
            term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_d_theta0')),
            term(-1.0, Q + ('a1s_bar', 'dCT_sigma_dtheta0')),
            term(-1.0, Q + ('i_M', 'dCT_sigma_dtheta0')),
        ],
        'dX_dA1': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dA1'))],
        'dX_dB1': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dB1'))],

        # ---- Y, pp. 579-580 --------------------------------------------
        'dY_dxdot': [
            term(1.0, Q + ('dCY_sigma_db1s', 'd_b1s_d_mu', 'd_mu_d_xdot')),
            term(1.0, Q + ('b1s_bar', 'dCT_sigma_dmu', 'd_mu_d_xdot')),
        ],
        # -rho A_b (Omega R)^2 [C_H_bar/sigma + (C_T_bar/sigma)(B_1 + a1s_bar)]
        #   dbeta/dydot -- the thrust vector tilted by B_1 + a1s_bar giving a
        # side force in sideslip. The subscript renders as an r in the scan;
        # it is T, which is also what the printed -14 requires.
        'dY_dydot': [
            term(-1.0, Q + ('CH_sigma_bar', 'd_beta_d_ydot')),
            term(-1.0, Q + ('CT_sigma_bar', 'B_1', 'd_beta_d_ydot')),
            term(-1.0, Q + ('CT_sigma_bar', 'a1s_bar', 'd_beta_d_ydot')),
        ],
        'dY_dzdot': [term(1.0, Q + ('d_b1s_d_lambda', 'dCY_sigma_db1s',
                                    'd_lambda_d_zdot'))],
        'dY_dq': [term(1.0, Q + ('dCY_sigma_db1s', 'd_b1s_dq')),
                  ref(1.0, 'dY_dxdot', ('h_M',))],
        'dY_dp': [term(1.0, Q + ('dCY_sigma_db1s', 'd_b1s_dp')),
                  ref(1.0, 'dY_dydot', ('h_M',))],
        'dY_dtheta0': [
            term(1.0, Q + ('dCY_sigma_db1s', 'd_b1s_d_theta0')),
            term(1.0, Q + ('b1s_bar', 'dCT_sigma_dtheta0')),
        ],
        'dY_dA1': [term(1.0, Q + ('dCY_sigma_db1s', 'd_b1s_dA1'))],
        'dY_dB1': [term(1.0, Q + ('dCY_sigma_db1s', 'd_b1s_dB1'))],

        # ---- Z, p. 580 -------------------------------------------------
        'dZ_dxdot': [
            term(-1.0, Q + ('dCT_sigma_dmu', 'd_mu_d_xdot')),
            term(-1.0, Q + ('dCT_sigma_dlambda', 'd_lambda_d_xdot')),
        ],
        'dZ_dzdot': [term(-1.0, Q + ('dCT_sigma_dlambda', 'd_lambda_d_zdot'))],
        # (2/Omega) rho A_b (Omega R)^2 C_T_bar/sigma
        'dZ_dr': [term(2.0, Q + ('CT_sigma_bar',), over=('Omega',))],
        'dZ_dtheta0': [term(-1.0, Q + ('dCT_sigma_dtheta0',))],
        # -rho A_b (Omega R)^2 (dCT/dlambda')(dlambda'/da1s)(da1s/dB1),
        # with dlambda'/da1s = mu: tilting the disc by a1s changes the inflow
        # through it by mu a1s.
        'dZ_dB1': [term(-1.0, Q + ('dCT_sigma_dlambda', 'mu', 'd_a1s_dB1'))],

        # ---- R, pp. 580-581 --------------------------------------------
        'dR_dxdot': [term(1.0, ('dR_db1s', 'd_b1s_d_mu', 'd_mu_d_xdot')),
                     ref(1.0, 'dY_dxdot', ('h_M',)),
                     ref(1.0, 'dZ_dxdot', ('y_M',))],
        # -(dR/db1s)(B_1 + a1s_bar)(dbeta/dydot) + (dY/dydot) h_M
        'dR_dydot': [term(-1.0, ('dR_db1s', 'B_1', 'd_beta_d_ydot')),
                     term(-1.0, ('dR_db1s', 'a1s_bar', 'd_beta_d_ydot')),
                     ref(1.0, 'dY_dydot', ('h_M',))],
        'dR_dzdot': [term(1.0, ('dR_db1s', 'd_b1s_d_lambda', 'd_lambda_d_zdot')),
                     ref(1.0, 'dY_dzdot', ('h_M',)),
                     ref(1.0, 'dZ_dzdot', ('y_M',))],
        'dR_dq': [term(1.0, ('dR_db1s', 'd_b1s_dq')),
                  ref(1.0, 'dY_dq', ('h_M',))],
        'dR_dp': [term(1.0, ('dR_db1s', 'd_b1s_dp')),
                  ref(1.0, 'dY_dp', ('h_M',))],
        'dR_dtheta0': [term(1.0, ('dR_db1s', 'd_b1s_d_theta0')),
                       ref(1.0, 'dY_dtheta0', ('h_M',)),
                       ref(1.0, 'dZ_dtheta0', ('y_M',))],
        'dR_dA1': [term(1.0, ('dR_db1s', 'd_b1s_dA1')),
                   ref(1.0, 'dY_dA1', ('h_M',))],
        'dR_dB1': [term(1.0, ('dR_db1s', 'd_b1s_dB1')),
                   ref(1.0, 'dY_dB1', ('h_M',))],

        # ---- M, p. 581 -------------------------------------------------
        'dM_dxdot': [term(1.0, ('dM_da1s', 'd_a1s_d_mu', 'd_mu_d_xdot')),
                     term(1.0, ('dM_da1s', 'd_a1s_d_lambda', 'd_lambda_d_xdot')),
                     ref(-1.0, 'dX_dxdot', ('h_M',)),
                     ref(1.0, 'dZ_dxdot', ('l_M',))],
        'dM_dydot': [ref(-1.0, 'dX_dydot', ('h_M',)),
                     term(1.0, ('dM_da1s', 'A_1', 'd_beta_d_ydot')),
                     term(-1.0, ('dM_da1s', 'b1s_bar', 'd_beta_d_ydot'))],
        'dM_dzdot': [term(1.0, ('dM_da1s', 'd_a1s_d_lambda', 'd_lambda_d_zdot')),
                     ref(-1.0, 'dX_dzdot', ('h_M',)),
                     ref(1.0, 'dZ_dzdot', ('l_M',))],
        'dM_dq': [term(1.0, ('dM_da1s', 'd_a1s_dq')),
                  ref(-1.0, 'dX_dq', ('h_M',))],
        'dM_dp': [term(1.0, ('dM_da1s', 'd_a1s_dp')),
                  ref(-1.0, 'dX_dp', ('h_M',))],
        'dM_dtheta0': [term(1.0, ('dM_da1s', 'd_a1s_d_theta0')),
                       ref(-1.0, 'dX_dtheta0', ('h_M',)),
                       ref(1.0, 'dZ_dtheta0', ('l_M',))],
        'dM_dA1': [term(1.0, ('dM_da1s', 'd_a1s_dA1')),
                   ref(-1.0, 'dX_dA1', ('h_M',))],
        'dM_dB1': [term(1.0, ('dM_da1s', 'd_a1s_dB1')),
                   ref(-1.0, 'dX_dB1', ('h_M',))],

        # ---- N, p. 582 -------------------------------------------------
        'dN_dxdot': [term(1.0, QR + ('dCQ_sigma_dmu', 'd_mu_d_xdot')),
                     term(1.0, QR + ('dCQ_sigma_dlambda', 'd_lambda_d_xdot'))],
        'dN_dzdot': [term(1.0, QR + ('dCQ_sigma_dlambda', 'd_lambda_d_zdot'))],
        # -(2/Omega) rho A_b (Omega R)^2 R C_Q_bar/sigma -- note the sign,
        # which Table 9.2 does not carry in hover. See C9-15.
        'dN_dr': [term(-2.0, QR + ('CQ_sigma_bar',), over=('Omega',))],
        'dN_dtheta0': [term(1.0, QR + ('dCQ_sigma_dtheta0',))],
    }


class MainRotorDerivativesFFComp(MonomialRowsComp):
    """Table 9.8: the main rotor's 41 dimensional derivatives at 115 knots.

    The largest table in the chapter, and the one the whole forward-flight
    analysis rests on. Rows are declared in the nested form the book prints
    them in -- ``dM/dq = (dM/da1s)(da1s/dq) - (dX/dq) h_M``, with ``dX/dq``
    itself two terms -- and ``MonomialRowsComp`` flattens the references.

    Where hover differs
    -------------------
    Three structural additions over Table 9.2.

    **The rate rows pick up a kinematic term.** ``dX/dq`` carries
    ``-(dX/dxdot) h_M`` and ``dX/dp`` carries ``-(dX/dydot) h_M``: a pitch rate
    puts a fore-and-aft velocity ``-q h_M`` at a hub sitting ``h_M`` above the
    c.g., and in forward flight that velocity changes the H-force. Table 9.2
    has no such term because in hover ``dX/dxdot`` is -5 against -12 here and
    the coupling is negligible.

    **``dlambda'/da1s = mu``.** Tilting the tip path plane by ``a1s`` in
    forward flight changes the inflow through it by ``mu a1s``, which is what
    makes ``dZ/dB1`` exist at all. In hover it is zero and the row is absent.

    **``dZ/dr`` and ``dN/dr`` are governed-engine terms**, ``2 Q/Omega`` in
    disguise, and p. 580 prints the ``2/Omega`` explicitly where Table 9.2
    omits it (entry C9-6).

    Three places the book disagrees with itself
    -------------------------------------------
    All three are documented in ``docs/validation_stability.md`` and the
    component follows the printed equations, not the printed values, except
    where noted.

    C9-14: ``da1s/dB1``. Table 9.6 prints -1.188 and its formula gives
    -1.18848. Table 9.8's ``dX/dB1 = 18,601`` and ``dM/dB1 = -364,158`` are
    both reproduced exactly by **-1.118**, a transposition, while
    ``dZ/dB1 = 67,855`` needs -1.188. The X and M columns and the Z column of
    the same table used different numbers, and the X value propagates into
    Tables 9.19 and 9.20.

    C9-15: ``dN/dr``. p. 582 carries a leading minus and p. 569 does not, for
    the same ``2 Q/Omega``. The forward-flight sign gives yaw damping; the
    hover sign gives yaw divergence, which is why the hover total needed the
    tail rotor to overcome the main rotor.

    C9-16: the middle term of ``dX/dzdot``. p. 578 prints ``C_T_bar/sigma``
    there, where ``dX/dxdot`` and ``dX/dtheta0`` print ``dCH/sigma/da1s`` in
    the same structural position. The two differ by ``(a/8)lambda' = -.017``,
    25 %, and the printed -6 needs ``C_T_bar/sigma`` -- with
    ``dCH/sigma/da1s`` the row comes out **+0.3**, a sign change. Equation and
    value agree within the row; both disagree with the rest of the column.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    Thirty-one of the forty-one rows land within one per cent of the printed
    value; see the validation notes for the full comparison and for the six
    trim quantities recovered from the table itself.
    """

    scalar_inputs = SCALAR_INPUTS
    input_units = INPUT_UNITS
    input_defaults = {'y_M': 0.0, 'rho': 0.002378, 'A_b': 240.0,
                      'Omega_R': 650.0, 'Omega': 650.0 / 30.0, 'R': 30.0,
                      'h_M': 7.5, 'l_M': -0.5, 'mu': 0.30, 'i_M': 0.0}

    def rows(self):
        return build_rows()
