"""Table 9.4 tests -- total derivatives in hover (pp. 571-573)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.main_rotor_derivatives_hover_comp import (
    ROWS as MAIN_ROWS,
)
from prouty.stability.tail_rotor_derivatives_hover_comp import build_rows
from prouty.stability.total_derivatives_hover_comp import (
    SHARED,
    TotalDerivativesHoverComp,
)

from test_g1_table92 import TABLE_9_2
from test_g1_table93 import TABLE_9_3

# Table 9.4, pp. 571-573: (main rotor, tail rotor, total). A blank cell in the
# book is None here. Transcribed row by row from the printed table.
TABLE_9_4 = {
    'dX_dxdot': (-5.0, None, -5.0),
    'dX_dydot': (-1.0, None, -1.0),
    'dX_dq': (1008.0, None, 1008.0),
    'dX_dp': (-355.0, None, -355.0),
    'dX_dtheta0_M': (3677.0, None, 3677.0),
    'dX_dA1': (-825.0, None, -825.0),
    'dX_dB1': (9531.0, None, 9531.0),
    'dY_dxdot': (1.0, None, 1.0),
    'dY_dydot': (-5.0, -13.0, -18.0),
    'dY_dq': (-355.0, None, -355.0),
    'dY_dp': (-1008.0, -78.0, -1086.0),
    'dY_dr': (None, 481.0, 481.0),
    'dY_dtheta0_M': (-3971.0, None, -3971.0),
    'dY_dA1': (9531.0, None, 9531.0),
    'dY_dB1': (825.0, None, 825.0),
    'dY_dtheta0_T': (None, 9746.0, 9746.0),
    'dZ_dzdot': (-182.0, None, -182.0),
    'dZ_dtheta0_M': (-147088.0, None, -147088.0),
    'dR_dxdot': (39.0, None, 39.0),
    'dR_dydot': (-143.0, 78.0, -65.0),
    'dR_dq': (-10097.0, None, -10097.0),
    'dR_dp': (-28659.0, -468.0, -29127.0),
    'dR_dr': (None, 2886.0, 2886.0),
    'dR_dtheta0_M': (-29783.0, None, -29783.0),
    'dR_dA1': (271016.0, None, 271016.0),
    'dR_dB1': (23468.0, None, 23468.0),
    'dR_dtheta0_T': (None, 58476.0, 58476.0),
    'dM_dxdot': (143.0, None, 143.0),
    'dM_dydot': (39.0, -7.0, 32.0),
    'dM_dzdot': (91.0, None, 91.0),
    'dM_dq': (-28659.0, None, -28659.0),
    'dM_dp': (10097.0, None, 10097.0),
    'dM_dr': (None, 274.0, 274.0),
    'dM_dtheta0_M': (45967.0, None, 45967.0),
    'dM_dA1': (23468.0, None, 23468.0),
    'dM_dB1': (-271016.0, None, -271016.0),
    'dM_dtheta0_T': (None, 11276.0, 11276.0),
    'dN_dydot': (None, 481.0, 481.0),
    'dN_dzdot': (-847.0, None, -847.0),
    'dN_dp': (None, 2886.0, 2886.0),
    'dN_dr': (4471.0, -17797.0, -13326.0),
    'dN_dtheta0_M': (564242.0, None, 564242.0),
    'dN_dtheta0_T': (None, -360602.0, -360602.0),
}

# The two totals that inherit a Table 9.3 correction. See C9-8 and C9-9.
CORRECTED = {'dR_dydot': -221.0, 'dM_dtheta0_T': -11276.0}


def run(nn=1):
    """Feed the printed Table 9.2 and 9.3 columns, and sum them."""
    prob = om.Problem()
    prob.model.add_subsystem('total', TotalDerivativesHoverComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)

    for name, (main, tail, _) in TABLE_9_4.items():
        stem = name[:-2] if name.endswith(('_M', '_T')) else name
        if main is not None:
            prob.set_val(f'{stem}_main', np.full(nn, main))
        if tail is not None:
            prob.set_val(f'{stem}_tail', np.full(nn, tail))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name, row', TABLE_9_4.items())
def test_table_9_4_totals(name, row):
    """Every printed total, from the printed component columns."""
    _, _, total = row
    assert_near_equal(run().get_val(name)[0], total, 1e-10)


def test_the_book_has_forty_three_rows_and_the_component_forty_four():
    """dR/dzdot is the one row Table 9.4 omits, because y_M = 0 makes it zero."""
    outputs = set(TotalDerivativesHoverComp().rows())
    assert len(TABLE_9_4) == 43
    assert outputs - set(TABLE_9_4) == {'dR_dzdot'}
    assert not set(TABLE_9_4) - outputs


def test_only_six_rows_take_both_rotors():
    """Every other row of Table 9.4 has a blank in one of its two columns."""
    both = {name for name, (main, tail, _) in TABLE_9_4.items()
            if main is not None and tail is not None}
    assert both == set(SHARED)


def test_collective_rows_are_never_summed():
    """theta_0M and theta_0T are separate controls, so they stay apart."""
    prob = run()
    assert_near_equal(prob.get_val('dY_dtheta0_M')[0], -3971.0, 1e-10)
    assert_near_equal(prob.get_val('dY_dtheta0_T')[0], 9746.0, 1e-10)


def test_every_source_row_is_consumed():
    """Nothing from Table 9.2 or 9.3 is dropped on the way into Table 9.4."""
    consumed = {name for terms in TotalDerivativesHoverComp().rows().values()
                for _, numerator, _ in terms for name in numerator}
    assert {f'{name}_main' for name in MAIN_ROWS} <= consumed
    assert {f'{name}_tail' for name in build_rows(1.0)} <= consumed


@pytest.mark.parametrize('name, expected', CORRECTED.items())
def test_totals_that_inherit_a_correction(name, expected):
    """C9-8 and C9-9 move two totals away from the printed ones."""
    main, tail, printed = TABLE_9_4[name]
    corrected = (main or 0.0) - (tail or 0.0)
    assert_near_equal(corrected, expected, 1e-10)
    assert abs(corrected - printed) > 1.0


def test_dihedral_effect_changes_sign_of_the_lateral_picture():
    """dR/dydot: -65 in the book, -221 once C9-8 is applied.

    Both are negative, but the corrected one is three times larger, and it
    feeds the hover lateral mode of p. 604.
    """
    printed = TABLE_9_4['dR_dydot'][2]
    assert printed < 0.0 and CORRECTED['dR_dydot'] < 0.0
    assert CORRECTED['dR_dydot'] / printed > 3.0


def test_source_tables_agree_with_the_table_9_4_columns():
    """The main and tail columns of Table 9.4 are Tables 9.2 and 9.3 verbatim."""
    for name, (main, tail, _) in TABLE_9_4.items():
        stem = name[:-2] if name.endswith(('_M', '_T')) else name
        if main is not None:
            assert TABLE_9_2[stem] == main, name
        if tail is not None:
            assert TABLE_9_3[stem] == tail, name


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_4:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-13)
