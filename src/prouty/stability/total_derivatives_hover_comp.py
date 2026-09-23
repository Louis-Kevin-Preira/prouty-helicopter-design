"""Total helicopter stability derivatives in hover.

Reference
---------
R. W. Prouty, *Helicopter Performance, Stability and Control*,
Chapter 9 "Stability and Control Analysis", Table 9.4, pp. 571-573. Sums the
main rotor's Table 9.2 (pp. 566-569) and the tail rotor's Table 9.3
(pp. 569-570).
"""

from prouty.stability.monomial_rows_comp import (
    DERIVATIVE_UNITS,
    MonomialRowsComp,
    term,
)

#: The six rows both rotors contribute to. Everything else is single-source.
SHARED = ('dY_dydot', 'dY_dp', 'dR_dydot', 'dR_dp', 'dM_dydot', 'dN_dr')

#: Table 9.2 rows the tail rotor has no counterpart for.
MAIN_ONLY = (
    'dX_dxdot', 'dX_dydot', 'dX_dq', 'dX_dp', 'dX_dA1', 'dX_dB1',
    'dY_dxdot', 'dY_dq', 'dY_dA1', 'dY_dB1',
    'dZ_dzdot',
    'dR_dxdot', 'dR_dzdot', 'dR_dq', 'dR_dA1', 'dR_dB1',
    'dM_dxdot', 'dM_dzdot', 'dM_dq', 'dM_dp', 'dM_dA1', 'dM_dB1',
    'dN_dzdot',
)

#: Table 9.3 rows the main rotor has no counterpart for.
TAIL_ONLY = ('dY_dr', 'dR_dr', 'dM_dr', 'dN_dydot', 'dN_dp')

#: Collective is a separate control on each rotor, so these rows are never
#: summed; Table 9.4 keeps them apart as theta_0M and theta_0T.
COLLECTIVE_MAIN = ('dX_dtheta0', 'dY_dtheta0', 'dZ_dtheta0', 'dR_dtheta0',
                   'dM_dtheta0', 'dN_dtheta0')
COLLECTIVE_TAIL = ('dY_dtheta0', 'dR_dtheta0', 'dM_dtheta0', 'dN_dtheta0')


def build_rows():
    """Table 9.4, as sums over the two rotors."""
    rows = {}
    for name in SHARED:
        rows[name] = [term(1.0, (f'{name}_main',)), term(1.0, (f'{name}_tail',))]
    for name in MAIN_ONLY:
        rows[name] = [term(1.0, (f'{name}_main',))]
    for name in TAIL_ONLY:
        rows[name] = [term(1.0, (f'{name}_tail',))]
    for name in COLLECTIVE_MAIN:
        rows[f'{name}_M'] = [term(1.0, (f'{name}_main',))]
    for name in COLLECTIVE_TAIL:
        rows[f'{name}_T'] = [term(1.0, (f'{name}_tail',))]
    return rows


#: Suffixes a derivative name can carry: the rotor a collective belongs to,
#: and the component a contribution comes from.
NAME_SUFFIXES = ('_M', '_T', '_main', '_tail', '_horiz', '_vert', '_fuse')


def derivative_units(name):
    """Units of ``dFORCE_dWRT``, tolerating any of ``NAME_SUFFIXES``."""
    stem = name
    for suffix in NAME_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    force, wrt = stem[1], stem.split('_d', 1)[1]
    units = DERIVATIVE_UNITS[wrt]
    return units.replace('lbf', 'lbf*ft') if force in 'RMN' else units


class TotalDerivativesHoverComp(MonomialRowsComp):
    """Table 9.4: main rotor plus tail rotor, the derivative set hover flies on.

    Forty-four outputs, from 35 main rotor inputs suffixed ``_main`` and 15
    tail rotor inputs suffixed ``_tail``. Only six rows actually add two
    contributions -- ``dY/dydot``, ``dY/dp``, ``dR/dydot``, ``dR/dp``,
    ``dM/dydot`` and ``dN/dr``. The rest pass a single rotor's value straight
    through, which is what the blank cells of Table 9.4 mean.

    Collective is not summed. Each rotor has its own, so the six main rotor
    collective rows come out as ``..._M`` and the four tail rotor ones as
    ``..._T``, matching the ``theta_0M`` and ``theta_0T`` of the printed table.

    Forty-four against the book's forty-three
    -----------------------------------------
    Table 9.4 has no ``dR/dzdot`` row. Table 9.2 does, as
    ``(dZ/dzdot) y_M = 0``, and it is zero only because the example
    helicopter's main rotor sits on the centreline. The row is kept here so a
    laterally offset rotor is not silently dropped.

    Where the totals depart from the book
    -------------------------------------
    ``dR/dydot``. Table 9.4 lists -143 main, +78 tail, -65 total. The tail
    rotor value contradicts its own equation in Table 9.3, which gives -79
    (entry C9-8 of ``docs/validation_stability.md``), so the total here is
    -222. This one matters: ``dR/dydot`` is the dihedral-effect derivative and
    it enters the hover lateral mode of p. 604, whose published roots were
    computed from -65.

    ``dM/dtheta0_T``. -11,276 here against the printed +11,276, from entry
    C9-9. It enters no hover analysis in the chapter.

    Both differences originate in ``TailRotorDerivativesHoverComp``; this
    component only adds up what it is given, and has no switch of its own.

    Options
    -------
    num_nodes : int

    Example helicopter
    ------------------
    Forty-one of the forty-three printed rows are reproduced, the two above
    excepted. Rows fed by the tail rotor sit about 1.5 % off where the tail
    rotor's own rounding puts them (see the Table 9.3 anchors).
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
