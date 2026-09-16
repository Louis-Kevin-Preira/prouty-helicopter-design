"""Main rotor dimensional stability derivatives near hover.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.2, pp. 566-569. Totals
cross-checked against Table 9.4, pp. 571-573. Inputs are the non-dimensional
derivatives of Table 9.1, pp. 564-565.
"""

from prouty.stability.monomial_rows_comp import MonomialRowsComp, term

#: rho A_b (Omega R)^2, the dynamic scale in front of every force derivative.
Q = ('rho', 'A_b', 'Omega_R', 'Omega_R')
#: rho A_b (Omega R)^2 R, the same for torque-derived moment derivatives.
QR = Q + ('R',)


# Every entry of Table 9.2, flattened into sums of monomials in the inputs.
# The comment on each row is the equation as printed; where the printed form
# refers to another derivative of the table, it has been substituted so that
# each row depends on inputs alone.
ROWS = {
    # ---- X, p. 566 -----------------------------------------------------
    # -rho A_b (Omega R)^2 (dCH/sigma/da1s)(da1s/dmu)(dmu/dxdot)
    'dX_dxdot': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_d_mu',
                                 'd_mu_d_xdot'))],
    # -(dY/dxdot)
    'dX_dydot': [term(-1.0, Q + ('dCY_sigma_db1s', 'd_b1s_d_mu',
                                 'd_mu_d_xdot'))],
    # -rho A_b (Omega R)^2 (dCH/sigma/da1s)(da1s/dq)
    'dX_dq': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dq'))],
    'dX_dp': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dp'))],
    # -rho A_b (Omega R)^2 (a1s_bar + i_M)(dCT/sigma/dtheta0)
    'dX_dtheta0': [term(-1.0, Q + ('a1s_bar', 'dCT_sigma_dtheta0')),
                   term(-1.0, Q + ('i_M', 'dCT_sigma_dtheta0'))],
    'dX_dA1': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dA1'))],
    'dX_dB1': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dB1'))],

    # ---- Y, pp. 566-567 ------------------------------------------------
    # +rho A_b (Omega R)^2 (dCY/sigma/db1s)(db1s/dmu)(dmu/dxdot)
    'dY_dxdot': [term(1.0, Q + ('dCY_sigma_db1s', 'd_b1s_d_mu',
                                'd_mu_d_xdot'))],
    'dY_dydot': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_d_mu',
                                 'd_mu_d_xdot'))],                  # = dX/dxdot
    'dY_dq': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dp'))],      # = dX/dp
    'dY_dp': [term(1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dq'))],       # = -dX/dq
    # +rho A_b (Omega R)^2 b1s_bar (dCT/sigma/dtheta0)
    'dY_dtheta0': [term(1.0, Q + ('b1s_bar', 'dCT_sigma_dtheta0'))],
    'dY_dA1': [term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dB1'))],    # = dX/dB1
    'dY_dB1': [term(1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dA1'))],     # = -dX/dA1

    # ---- Z, p. 567 -----------------------------------------------------
    # -rho A_b (Omega R)^2 (dCT/sigma/dlambda')(dlambda'/dzdot)
    'dZ_dzdot': [term(-1.0, Q + ('dCT_sigma_dlambda', 'd_lambda_d_zdot'))],
    'dZ_dtheta0': [term(-1.0, Q + ('dCT_sigma_dtheta0',))],

    # ---- R, pp. 567-568 ------------------------------------------------
    # (dR/db1s)(db1s/dmu)(dmu/dxdot) + (dY/dxdot) h_M
    'dR_dxdot': [term(1.0, ('dR_db1s', 'd_b1s_d_mu', 'd_mu_d_xdot')),
                 term(1.0, Q + ('dCY_sigma_db1s', 'd_b1s_d_mu',
                                'd_mu_d_xdot', 'h_M'))],
    # -(dM/dxdot)
    'dR_dydot': [term(-1.0, ('dM_da1s', 'd_a1s_d_mu', 'd_mu_d_xdot')),
                 term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_d_mu',
                                 'd_mu_d_xdot', 'h_M'))],
    # (dZ/dzdot) y_M
    'dR_dzdot': [term(-1.0, Q + ('dCT_sigma_dlambda', 'd_lambda_d_zdot',
                                 'y_M'))],
    # (dR/db1s)(db1s/dq) + (dY/dq) h_M
    'dR_dq': [term(1.0, ('dR_db1s', 'd_b1s_dq')),
              term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dp', 'h_M'))],
    'dR_dp': [term(1.0, ('dR_db1s', 'd_b1s_dp')),
              term(1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dq', 'h_M'))],
    # (dY/dtheta0) h_M + (dZ/dtheta0) y_M
    'dR_dtheta0': [term(1.0, Q + ('b1s_bar', 'dCT_sigma_dtheta0', 'h_M')),
                   term(-1.0, Q + ('dCT_sigma_dtheta0', 'y_M'))],
    # (dR/db1s)(db1s/dA1) + (dY/dA1) h_M
    'dR_dA1': [term(1.0, ('dR_db1s', 'd_b1s_dA1')),
               term(-1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dB1', 'h_M'))],
    'dR_dB1': [term(1.0, ('dR_db1s', 'd_b1s_dB1')),
               term(1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dA1', 'h_M'))],

    # ---- M, p. 568 -----------------------------------------------------
    # (dM/da1s)(da1s/dmu)(dmu/dxdot) - (dX/dxdot) h_M
    'dM_dxdot': [term(1.0, ('dM_da1s', 'd_a1s_d_mu', 'd_mu_d_xdot')),
                 term(1.0, Q + ('dCH_sigma_da1s', 'd_a1s_d_mu',
                                'd_mu_d_xdot', 'h_M'))],
    # (dR/dxdot)
    'dM_dydot': [term(1.0, ('dR_db1s', 'd_b1s_d_mu', 'd_mu_d_xdot')),
                 term(1.0, Q + ('dCY_sigma_db1s', 'd_b1s_d_mu',
                                'd_mu_d_xdot', 'h_M'))],
    # (dZ/dzdot) l_M
    'dM_dzdot': [term(-1.0, Q + ('dCT_sigma_dlambda', 'd_lambda_d_zdot',
                                 'l_M'))],
    # (dM/da1s)(da1s/dq) - (dX/dq) h_M
    'dM_dq': [term(1.0, ('dM_da1s', 'd_a1s_dq')),
              term(1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dq', 'h_M'))],
    'dM_dp': [term(1.0, ('dM_da1s', 'd_a1s_dp')),
              term(1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dp', 'h_M'))],
    # -(dX/dtheta0) h_M + (dZ/dtheta0) l_M
    'dM_dtheta0': [term(1.0, Q + ('a1s_bar', 'dCT_sigma_dtheta0', 'h_M')),
                   term(1.0, Q + ('i_M', 'dCT_sigma_dtheta0', 'h_M')),
                   term(-1.0, Q + ('dCT_sigma_dtheta0', 'l_M'))],
    'dM_dA1': [term(1.0, ('dM_da1s', 'd_a1s_dA1')),
               term(1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dA1', 'h_M'))],
    'dM_dB1': [term(1.0, ('dM_da1s', 'd_a1s_dB1')),
               term(1.0, Q + ('dCH_sigma_da1s', 'd_a1s_dB1', 'h_M'))],

    # ---- N, pp. 568-569 ------------------------------------------------
    # rho A_b (Omega R)^2 R (dCQ/sigma/dlambda')(dlambda'/dzdot)
    'dN_dzdot': [term(1.0, QR + ('dCQ_sigma_dlambda', 'd_lambda_d_zdot'))],
    # 2 rho A_b (Omega R)^2 R (CQ_bar/sigma) / Omega
    'dN_dr': [term(2.0, QR + ('CQ_sigma_bar',), over=('Omega',))],
    'dN_dtheta0': [term(1.0, QR + ('dCQ_sigma_dtheta0',))],
}

SCALAR_INPUTS = ('A_b', 'R', 'h_M', 'l_M', 'y_M', 'i_M')

#: Must match the units the Table 9.1 component declares on its outputs.
INPUT_UNITS = {
    'rho': 'slug/ft**3', 'A_b': 'ft**2', 'Omega_R': 'ft/s', 'Omega': 'rad/s',
    'R': 'ft', 'h_M': 'ft', 'l_M': 'ft', 'y_M': 'ft',
    'd_mu_d_xdot': 's/ft', 'd_lambda_d_zdot': 's/ft',
    'd_a1s_dq': 's', 'd_a1s_dp': 's', 'd_b1s_dq': 's', 'd_b1s_dp': 's',
    'dM_da1s': 'lbf*ft/rad', 'dR_db1s': 'lbf*ft/rad',
    'a1s_bar': 'rad', 'b1s_bar': 'rad', 'i_M': 'rad',
}


class MainRotorDerivativesHoverComp(MonomialRowsComp):
    """Table 9.2: the main rotor's 35 dimensional derivatives in hover.

    Six forces and moments (X, Y, Z, R, M, N) against the velocities, rates
    and controls that move them near hover. Only the combinations Prouty
    prints are produced; the ones he leaves blank are identically zero in
    hover and would only add rows of zeros to Table 9.4.

    How the rows are stored
    -----------------------
    Every entry of Table 9.2 is a sum of monomials in the Table 9.1
    derivatives, the dynamic scale ``rho A_b (Omega R)^2`` and the hub offsets,
    so the table is transcribed once, declaratively, in ``ROWS``, and
    ``MonomialRowsComp`` derives the 35 outputs and their roughly 200 partials
    from it.

    Rows that the book writes in terms of *other* rows -- ``dR/dq`` is
    ``(dR/db1s)(db1s/dq) + (dY/dq) h_M`` -- have that reference substituted so
    each row depends on inputs alone. The printed form is kept as a comment
    above every substituted row.

    Geometry the table implies
    --------------------------
    Table 9.2 never prints the hub offsets, but four rows pin them exactly:
    ``dM/dzdot = (dZ/dzdot) l_M`` gives ``l_M = -0.5 ft``, ``dR/dzdot = 0``
    gives ``y_M = 0``, and ``dR/dp`` and ``dR/dtheta0`` both give
    ``h_M = 7.5 ft``.

    Trim quantities
    ---------------
    Three trim values enter, and Table 9.2 only determines two of them
    directly: ``b1s_bar = -.027`` from ``dY/dtheta0`` and
    ``CQ_bar/sigma = .0067`` from ``dN/dr``. ``dX/dtheta0`` fixes only the
    *sum* ``a1s_bar + i_M = -.025``, so the two are separate inputs here and
    the split has to come from the Chapter 8 trim solution.

    Two readings worth recording
    ----------------------------
    p. 568 prints ``dM/dtheta0 = -(dX_A/dtheta0)_M h_M + (dZ/dtheta0)_M l_M``,
    with a subscript ``A`` on ``X`` that appears nowhere else in the table.
    It has no numerical consequence: with ``dX/dtheta0 = 3,677`` the row
    returns 45,966 against the printed 45,967.

    p. 569 prints ``dN/dr = 2 rho A_b (Omega R)^2 R C_Q/sigma`` with no
    ``Omega``. That expression is a moment, not a moment per unit rate, and it
    is 22 times the printed 4,471. The ``1/Omega`` is restored here; see entry
    C9-6 of ``docs/validation_stability.md``. It makes the row the familiar
    ``2 Q_M / Omega``: a governed engine holding ``Omega`` fixed, with torque
    going as ``Omega^2``.

    Options
    -------
    num_nodes : int

    Example helicopter
    ------------------
    Fed the printed Table 9.1 values with ``dCH/sigma/da1s = .0398`` (the
    unrounded value four rows of Table 9.2 imply, printed as .040), and
    ``h_M = 7.5``, ``l_M = -0.5``, ``y_M = 0``, all 35 rows land within a unit
    of the last printed digit, except ``dR/dxdot`` and ``dM/dydot``: 42.0
    against a printed 39, which is Prouty rounding ``dY/dxdot`` from 1.48 to 1
    before multiplying by ``h_M``.
    """

    scalar_inputs = SCALAR_INPUTS
    input_units = INPUT_UNITS
    input_defaults = {'y_M': 0.0}

    def rows(self):
        return ROWS
