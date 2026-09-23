"""Horizontal stabilizer dimensional derivatives in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.11, pp. 585-586. The
angle-of-attack derivatives it consumes are Table 9.10, p. 584; the steady
lift and drag behind the two brackets are Chapter 8, pp. 488-491.
"""

import numpy as np

from prouty.stability.monomial_rows_comp import MonomialRowsComp, ref, term

#: (q_H/q) q A_H a_H, the scale in front of both aerodynamic brackets.
S = ('qH_q', 'q', 'A_H', 'a_H')

SCALAR_INPUTS = ('A_H', 'a_H', 'A_R', 'alpha_LO', 'i_H', 'delta', 'C_D0',
                 'l_H', 'h_H')

INPUT_UNITS = {
    'q': 'lbf/ft**2', 'A_H': 'ft**2', 'a_H': '1/rad', 'V': 'ft/s',
    'alpha_H_bar': 'rad', 'alpha_LO': 'rad', 'i_H': 'rad',
    'X_H_bar': 'lbf', 'Z_H_bar': 'lbf', 'l_H': 'ft', 'h_H': 'ft',
    'd_alphaH_d_xdot': 'rad*s/ft', 'd_alphaH_d_zdot': 'rad*s/ft',
    'd_alphaH_d_zddot': 'rad*s**2/ft',
}

INPUT_DEFAULTS = {
    'qH_q': 0.6, 'q': 44.793, 'A_H': 18.0, 'a_H': 4.0, 'A_R': 4.5,
    'delta': 0.02, 'C_D0': 0.0064, 'alpha_LO': 0.0, 'i_H': -0.052,
    'alpha_H_bar': -0.1401, 'V': 194.1, 'X_H_bar': -14.06, 'Z_H_bar': 271.0,
    'l_H': 33.12, 'h_H': -2.968,
    'd_alphaH_d_xdot': 0.000761, 'd_alphaH_d_zdot': 0.003299,
    'd_alphaH_d_zddot': -0.000132,
}


def lift_bracket(coefficient, factors):
    """``(alpha_H - alpha_LO)[1 - 2 a_H (1+delta)/(pi A.R.)] + (alpha_H - i_H)``.

    The X-force bracket of pp. 585. Expanded into monomials so the component
    gets its partials for free; the printed form is the docstring above.
    """
    f, c = tuple(factors), coefficient
    induced = 2.0 * c / np.pi
    return [
        term(2.0 * c, f + ('alpha_H_bar',)),
        term(-c, f + ('alpha_LO',)),
        term(-c, f + ('i_H',)),
        term(-induced, f + ('a_H', 'alpha_H_bar'), over=('A_R',)),
        term(-induced, f + ('a_H', 'delta', 'alpha_H_bar'), over=('A_R',)),
        term(induced, f + ('a_H', 'alpha_LO'), over=('A_R',)),
        term(induced, f + ('a_H', 'delta', 'alpha_LO'), over=('A_R',)),
    ]


def drag_bracket(coefficient, factors):
    """``1 + [a_H(1+delta)/(pi A.R.)][2(a-aLO)(a-i_H) + (a-aLO)^2] + C_D0``.

    The Z-force bracket of pp. 585. The inner square bracket expands to
    ``3 a^2 - 2 a i_H - 4 aLO a + 2 aLO i_H + aLO^2``.
    """
    f, c = tuple(factors), coefficient
    induced = c / np.pi
    rows = [term(c, f), term(c, f + ('C_D0',))]
    inner = ((3.0, ('alpha_H_bar', 'alpha_H_bar')),
             (-2.0, ('alpha_H_bar', 'i_H')),
             (-4.0, ('alpha_LO', 'alpha_H_bar')),
             (2.0, ('alpha_LO', 'i_H')),
             (1.0, ('alpha_LO', 'alpha_LO')))
    for weight, names in inner:
        rows.append(term(induced * weight, f + ('a_H',) + names,
                         over=('A_R',)))
        rows.append(term(induced * weight, f + ('a_H', 'delta') + names,
                         over=('A_R',)))
    return rows


def build_rows():
    """Table 9.11, pp. 585-586."""
    lag = dict(factors=('d_alphaH_d_zddot',), over=('d_alphaH_d_zdot',))
    return {
        # (2/V) X_H_bar + (q_H/q) q A_H a_H {lift bracket} dalpha_H/dxdot
        'dX_dxdot': ([term(2.0, ('X_H_bar',), over=('V',))]
                     + lift_bracket(1.0, S + ('d_alphaH_d_xdot',))),
        'dX_dzdot': lift_bracket(1.0, S + ('d_alphaH_d_zdot',)),
        # (dX/dzdot)(dalpha_H/dzddot)/(dalpha_H/dzdot)
        'dX_dzddot': [ref(1.0, 'dX_dzdot', **lag)],

        # (2/V) Z_H_bar - (q_H/q) q A_H a_H {drag bracket} dalpha_H/dxdot
        'dZ_dxdot': ([term(2.0, ('Z_H_bar',), over=('V',))]
                     + drag_bracket(-1.0, S + ('d_alphaH_d_xdot',))),
        'dZ_dzdot': drag_bracket(-1.0, S + ('d_alphaH_d_zdot',)),
        'dZ_dzddot': [ref(1.0, 'dZ_dzdot', **lag)],
        'dZ_dq': [ref(1.0, 'dZ_dzdot', ('l_H',))],

        # -(dX/d.) h_H + (dZ/d.) l_H
        'dM_dxdot': [ref(-1.0, 'dX_dxdot', ('h_H',)),
                     ref(1.0, 'dZ_dxdot', ('l_H',))],
        'dM_dzdot': [ref(-1.0, 'dX_dzdot', ('h_H',)),
                     ref(1.0, 'dZ_dzdot', ('l_H',))],
        'dM_dzddot': [ref(-1.0, 'dX_dzddot', ('h_H',)),
                      ref(1.0, 'dZ_dzddot', ('l_H',))],
        'dM_dq': [ref(1.0, 'dZ_dq', ('l_H',))],
    }


class HorizStabDerivativesFFComp(MonomialRowsComp):
    """Table 9.11: the horizontal stabilizer's 11 dimensional derivatives.

    Two aerodynamic brackets do all the work. Everything else is those two
    carried onto the arms::

        lift bracket  (alpha_H - alpha_LO)[1 - 2 a_H (1+delta)/(pi A.R.)]
                      + (alpha_H - i_H)
        drag bracket  1 + [a_H(1+delta)/(pi A.R.)]
                      [2(alpha_H - alpha_LO)(alpha_H - i_H)
                       + (alpha_H - alpha_LO)^2] + C_D0

    The first is the slope of the chordwise force with angle of attack, the
    second the slope of the normal force. Both are expanded into monomials
    here so that the partials come out of the declaration; the printed forms
    are the docstrings of :func:`lift_bracket` and :func:`drag_bracket`.

    The ``2/V`` terms
    -----------------
    ``dX/dxdot`` and ``dZ/dxdot`` each carry ``(2/V)`` times the trim force.
    That is dynamic pressure, not angle of attack: forward speed changes ``q``
    as ``V^2``, so the force it already carries grows as ``2/V``. It is the
    larger part of ``dZ/dxdot`` -- 2.79 of the 1.29 total, against -1.50 from
    the angle-of-attack term.

    The lag rows
    ------------
    ``dX/dzddot`` and ``dZ/dzddot`` are the ``zdot`` rows scaled by
    ``(dalpha_H/dzddot)/(dalpha_H/dzdot)``, which is the downwash lag of
    Table 9.10 -- the only unsteady effect in the table. The ratio is -0.040
    for the example helicopter, so these rows are small but not zero, and
    ``dM/dzddot = 9`` is what puts the ``9 s^2`` into Table 9.20's M row
    (p. 614).

    The division means ``dalpha_H/dzdot`` must not be zero. It is .0033 here
    and is dominated by ``-1/V``, so it only vanishes at infinite airspeed.

    One printing artefact
    ---------------------
    The two Z rows must share the drag bracket, and the ``dZ/dxdot`` one on
    p. 585 prints ``(alpha_H - alpha_LO)^2`` clearly. The ``dZ/dzdot`` one has
    its minus sign broken by the scan and reads as ``(alpha_H alpha_LO)^2``.
    No consequence -- both rows use the same bracket.

    Options
    -------
    num_nodes : int

    Example helicopter at 115 knots
    -------------------------------
    =========== ========= ==========
    row         model     Table 9.11
    =========== ========= ==========
    dX_dxdot    -.362     <1
    dX_dzdot    -.941     -1
    dX_dzddot   .038      <1
    dZ_dxdot    1.292     1
    dZ_dzdot    -6.506    -7
    dZ_dzddot   .260      <1
    dZ_dq       -215.5    -217
    dM_dxdot    41.7      42
    dM_dzdot    -218.3    -219
    dM_dzddot   8.73      9
    dM_dq       -7,137    -7,161
    =========== ========= ==========

    The stabilizer's own parameters are the Chapter 8 ones already in this
    repository (Table 8.5, p. 523): ``q_H/q = .6``, ``A_H = 18 ft^2``,
    ``a_H = 4.0``, ``A.R. = 4.5``, ``delta = .02``, ``C_D0 = .0064``,
    ``i_H = -.052 rad``, ``alpha_H = -7.9 deg``. The two arms and the two trim
    forces are recovered from Table 9.11 itself; see the validation notes.
    """

    scalar_inputs = SCALAR_INPUTS
    input_units = INPUT_UNITS
    input_defaults = INPUT_DEFAULTS

    def rows(self):
        return build_rows()
