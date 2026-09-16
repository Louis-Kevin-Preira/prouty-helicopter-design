"""Total helicopter stability derivatives in forward flight.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.16, pp. 591-595. Sums the
main rotor (Table 9.8), tail rotor (Table 9.9), horizontal stabilizer
(Table 9.11), vertical stabilizer (Table 9.13) and fuselage (Table 9.15).
Totals cross-checked against Table 9.20, p. 614.
"""

from prouty.stability.monomial_rows_comp import MonomialRowsComp, term
from prouty.stability.total_derivatives_hover_comp import derivative_units

#: Input suffix for each of the five contributors, in Table 9.16's column
#: order.
COMPONENTS = ('main', 'tail', 'horiz', 'vert', 'fuse')

#: Which columns of Table 9.16 carry a number on each row, pp. 591-595.
CONTRIBUTORS = {
    # ---- X, pp. 591-592 ------------------------------------------------
    'dX_dxdot': ('main', 'fuse'),
    'dX_dydot': ('main', 'vert'),
    'dX_dzdot': ('main', 'horiz', 'fuse'),
    'dX_dq': ('main',),
    'dX_dp': ('main',),
    'dX_dtheta0_M': ('main',),
    'dX_dA1': ('main',),
    'dX_dB1': ('main',),

    # ---- Y, pp. 592-593 ------------------------------------------------
    'dY_dxdot': ('main', 'tail', 'vert'),
    'dY_dydot': ('main', 'tail', 'vert', 'fuse'),
    'dY_dzdot': ('main',),
    'dY_dq': ('main',),
    'dY_dp': ('main', 'tail', 'vert'),
    'dY_dr': ('tail', 'vert'),
    'dY_dtheta0_M': ('main',),
    'dY_dA1': ('main',),
    'dY_dB1': ('main',),
    'dY_dtheta0_T': ('tail',),

    # ---- Z, p. 593 -----------------------------------------------------
    'dZ_dxdot': ('main', 'horiz', 'fuse'),
    'dZ_dzdot': ('main', 'horiz', 'fuse'),
    'dZ_dq': ('horiz',),
    'dZ_dr': ('main',),
    'dZ_dtheta0_M': ('main',),
    'dZ_dB1': ('main',),

    # ---- R, pp. 593-594 ------------------------------------------------
    'dR_dxdot': ('main', 'tail', 'vert'),
    'dR_dydot': ('main', 'tail', 'vert', 'fuse'),
    'dR_dzdot': ('main',),
    'dR_dq': ('main',),
    'dR_dp': ('main', 'tail', 'vert'),
    'dR_dr': ('tail', 'vert'),
    'dR_dtheta0_M': ('main',),
    'dR_dA1': ('main',),
    'dR_dB1': ('main',),
    'dR_dtheta0_T': ('tail',),

    # ---- M, pp. 594-595 ------------------------------------------------
    'dM_dxdot': ('main', 'horiz', 'fuse'),
    'dM_dydot': ('main',),
    'dM_dzdot': ('main', 'horiz', 'fuse'),
    'dM_dzddot': ('horiz',),
    'dM_dq': ('main', 'horiz'),
    'dM_dp': ('main',),
    'dM_dtheta0_M': ('main',),
    'dM_dA1': ('main',),
    'dM_dB1': ('main',),

    # ---- N, p. 595 -----------------------------------------------------
    'dN_dxdot': ('main', 'tail', 'vert'),
    'dN_dydot': ('tail', 'vert', 'fuse'),
    'dN_dzdot': ('main',),
    'dN_dp': ('tail', 'vert'),
    'dN_dr': ('main', 'tail', 'vert'),
    'dN_dtheta0_M': ('main',),
    'dN_dtheta0_T': ('tail',),
}


def build_rows():
    """Table 9.16, as sums over the columns that carry a number."""
    rows = {}
    for name, sources in CONTRIBUTORS.items():
        stem = name[:-2] if name.endswith(('_M', '_T')) else name
        rows[name] = [term(1.0, (f'{stem}_{source}',)) for source in sources]
    return rows


class TotalDerivativesFFComp(MonomialRowsComp):
    """Table 9.16: five contributors, fifty rows, at 115 knots.

    The forward-flight counterpart of Table 9.4, and four times the size: the
    airframe is in it. A blank cell in the printed table means the component
    does not contribute to that row at all, so only the columns Prouty fills
    become inputs. Twenty-five of the fifty rows are main rotor alone.

    Collective is not summed, as in hover. The main rotor has six collective
    rows and the tail rotor three -- there is no ``dZ/dtheta0_T`` or
    ``dM/dtheta0_T``, because Table 9.9 drops the tail rotor's M rows
    altogether (entry C9-9 in the validation notes).

    What the totals are worth checking against
    ------------------------------------------
    Ten of them appear verbatim in Table 9.20, p. 614, which is the matrix the
    forward-flight stability analysis of pp. 617-634 is built on::

        dM/dxdot   144        dR/dp     -33,738
        dM/dzdot   650        dR/dr       6,912
        dM/dzddot    9        dN/dxdot     -136
        dM/dq  -43,752        dN/dydot    1,207
                              dN/dp       6,909
                              dN/dr     -53,913

    Those are the same numbers this repository's ``PolyDeterminantComp`` was
    first validated against, back at the top of the chapter. The derivative
    chain and the equations of motion close on each other.

    Where the totals inherit a disagreement
    ---------------------------------------
    Three rows carry a value the book contradicts elsewhere, and the totals
    carry it too. ``dX/dB1`` and ``dM/dB1`` are main rotor alone and inherit
    the transposed ``da1s/dB1 = -1.118`` of entry C9-14. ``dX/dydot`` takes
    the vertical stabilizer's -3, which entry C9-17 shows cannot be reached
    from the fin's own geometry.

    This component only adds up what it is given; none of the three
    originates here.

    Options
    -------
    num_nodes : int
    """

    def rows(self):
        return build_rows()

    @property
    def input_units(self):
        return {name: derivative_units(name)
                for terms in self.rows().values()
                for _, numerator, _ in terms
                for name in numerator}

    @staticmethod
    def units_of(name):
        return derivative_units(name)
