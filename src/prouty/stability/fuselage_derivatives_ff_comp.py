"""Fuselage dimensional derivatives in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.15, pp. 590-591. The
angle-of-attack derivatives are Table 9.14, pp. 589-590; the wind tunnel
slopes are Appendix A read at trim.
"""

from prouty.stability.monomial_rows_comp import MonomialRowsComp, term

SCALAR_INPUTS = ('df_dalphaF', 'dLq_dalphaF', 'dSFq_dbeta', 'dMq_dalphaF',
                 'dNq_dbeta', 'dRq_dbeta')

INPUT_UNITS = {
    'V': 'ft/s', 'q': 'lbf/ft**2',
    'X_F_bar': 'lbf', 'Z_F_bar': 'lbf', 'L_F_bar': 'lbf', 'D_F_bar': 'lbf',
    'M_F_bar': 'lbf*ft',
    'df_dalphaF': 'ft**2/rad', 'dLq_dalphaF': 'ft**2/rad',
    'dSFq_dbeta': 'ft**2/rad', 'dMq_dalphaF': 'ft**3/rad',
    'dNq_dbeta': 'ft**3/rad', 'dRq_dbeta': 'ft**3/rad',
    'd_alphaF_d_xdot': 'rad*s/ft', 'd_alphaF_d_zdot': 'rad*s/ft',
    'd_beta_d_ydot': 'rad*s/ft',
}

INPUT_DEFAULTS = {
    'V': 194.1, 'q': 44.793,
    'X_F_bar': -794.0, 'Z_F_bar': 281.0, 'L_F_bar': -281.0, 'D_F_bar': 794.0,
    'M_F_bar': -11733.0,
    'df_dalphaF': -2.0, 'dLq_dalphaF': 74.5, 'dSFq_dbeta': -220.0,
    'dMq_dalphaF': 1780.0, 'dNq_dbeta': -820.0, 'dRq_dbeta': 230.0,
    'd_alphaF_d_xdot': 0.000507, 'd_alphaF_d_zdot': 0.004637,
    'd_beta_d_ydot': 0.005152,
}


def build_rows():
    """Table 9.15, pp. 590-591."""
    return {
        # (2/V) X_F_bar
        'dX_dxdot': [term(2.0, ('X_F_bar',), over=('V',))],
        # (L_F_bar - q df/dalpha_F) dalpha_F/dzdot
        'dX_dzdot': [term(1.0, ('L_F_bar', 'd_alphaF_d_zdot')),
                     term(-1.0, ('q', 'df_dalphaF', 'd_alphaF_d_zdot'))],
        # (1/V)[q d(S.F./q)/dbeta - D_F_bar]
        'dY_dydot': [term(1.0, ('q', 'dSFq_dbeta'), over=('V',)),
                     term(-1.0, ('D_F_bar',), over=('V',))],
        'dZ_dxdot': [term(2.0, ('Z_F_bar',), over=('V',))],
        # (-D_F_bar - q d(L/q)/dalpha_F) dalpha_F/dzdot
        'dZ_dzdot': [term(-1.0, ('D_F_bar', 'd_alphaF_d_zdot')),
                     term(-1.0, ('q', 'dLq_dalphaF', 'd_alphaF_d_zdot'))],
        'dR_dydot': [term(1.0, ('q', 'dRq_dbeta', 'd_beta_d_ydot'))],
        # (2/V) M_F_bar + q d(M/q)/dalpha_F dalpha_F/dxdot
        'dM_dxdot': [term(2.0, ('M_F_bar',), over=('V',)),
                     term(1.0, ('q', 'dMq_dalphaF', 'd_alphaF_d_xdot'))],
        'dM_dzdot': [term(1.0, ('q', 'dMq_dalphaF', 'd_alphaF_d_zdot'))],
        'dN_dydot': [term(1.0, ('q', 'dNq_dbeta', 'd_beta_d_ydot'))],
    }


class FuselageDerivativesFFComp(MonomialRowsComp):
    """Table 9.15: the fuselage's nine dimensional derivatives.

    Every row is a wind tunnel slope times dynamic pressure times an
    angle-of-attack derivative, or a trim force times ``2/V``. There is no
    bracket and no feedback anywhere, which makes this the plainest of the
    three airframe tables -- and the only one whose aerodynamic content comes
    entirely from measurement rather than from a lift curve slope.

    Two kinds of row
    ----------------
    The ``2/V`` rows -- ``dX/dxdot``, ``dZ/dxdot`` and the first half of
    ``dM/dxdot`` -- are dynamic pressure alone: forward speed changes ``q`` as
    ``V^2``, so whatever force the fuselage already carries grows as ``2/V``.
    They need no Appendix A curve at all.

    The rest are ``q`` times a curve slope times an angle derivative. Note
    that ``dX/dzdot`` uses ``df/dalpha_F``, the equivalent flat plate area
    slope, where ``dZ/dzdot`` uses ``d(L/q)/dalpha_F``: chordwise force
    follows the drag area and normal force follows the lift area, and the two
    are separate curves.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    =========== ========= ==========
    row         model     Table 9.15
    =========== ========= ==========
    dX_dxdot    -8.18     -8
    dX_dzdot    -.888     -1
    dY_dydot    -54.9     -55
    dZ_dxdot    2.90      3
    dZ_dzdot    -19.16    -19
    dR_dydot    53.1      53
    dM_dxdot    -80.5     -80
    dM_dzdot    369.7     374
    dN_dydot    -189.2    -190
    =========== ========= ==========

    The trim forces are the Chapter 8 ones already in this repository:
    ``L_F_bar = -281 lb`` and ``D_F_bar = 794 lb``, Table 8.5 p. 524, with
    ``X_F_bar = -D_F_bar`` and ``Z_F_bar = -L_F_bar``. Only ``M_F_bar`` had to
    be recovered, -11,733 ft-lb from ``dM/dxdot = -80``.
    """

    scalar_inputs = SCALAR_INPUTS
    input_units = INPUT_UNITS
    input_defaults = INPUT_DEFAULTS

    def rows(self):
        return build_rows()
