"""G3 tests -- Table 9.13, vertical stabilizer derivatives in forward flight."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.vert_stab_derivatives_ff_comp import (
    DEFAULTS,
    DERIVED,
    SCALAR_INPUTS,
    VertStabDerivativesFFComp,
)

# Table 9.13, pp. 587-589. dX/dxdot is printed only as "<1".
TABLE_9_13 = {
    'dX_dydot': -3.0, 'dY_dxdot': 5.0, 'dY_dydot': -14.0,
    'dY_dp': -42.0, 'dY_dr': 490.0,
    'dR_dxdot': 12.0, 'dR_dydot': -42.0, 'dR_dp': -126.0, 'dR_dr': 1470.0,
    'dN_dxdot': -161.0, 'dN_dydot': 490.0, 'dN_dp': 1470.0, 'dN_dr': -17150.0,
}

#: Two rows the book contradicts itself on; see the tests below.
CONTRADICTED = ('dX_dydot', 'dR_dxdot')


def run(nn=1, **overrides):
    values = dict(DEFAULTS, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('v', VertStabDerivativesFFComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALAR_INPUTS else np.full(nn, value))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name, expected', TABLE_9_13.items())
def test_table_9_13(name, expected):
    if name in CONTRADICTED:
        pytest.skip('covered by its own test')
    got = run().get_val(name)[0]
    tol = 9e-2 if name == 'dY_dxdot' else 2.5e-2      # 4.58 prints as 5
    assert abs(got / expected - 1.0) <= tol, f'{name}: {got} vs {expected}'


def test_dX_dxdot_is_under_one():
    """The book prints "<1 lb/ft/sec"; the model gives .28."""
    assert 0.0 < run().get_val('dX_dxdot')[0] < 1.0


def test_the_table_has_fourteen_rows():
    assert len(TABLE_9_13) + 1 == len(DERIVED) + 4 == 14


def test_ten_rows_are_two_carried_on_the_arms():
    prob = run()
    h, l = DEFAULTS['h_V'], DEFAULTS['l_V']
    assert_near_equal(prob.get_val('dR_dydot')[0],
                      prob.get_val('dY_dydot')[0] * h, 1e-12)
    assert_near_equal(prob.get_val('dN_dydot')[0],
                      -prob.get_val('dY_dydot')[0] * l, 1e-12)
    assert_near_equal(prob.get_val('dN_dr')[0],
                      prob.get_val('dY_dydot')[0] * l ** 2, 1e-12)
    assert_near_equal(prob.get_val('dR_dp')[0],
                      prob.get_val('dY_dydot')[0] * h ** 2, 1e-12)


def test_the_arms_are_each_confirmed_three_ways():
    """h_V = 3.0 and l_V = 35.0, from the R and N rows of p. 588 and p. 589."""
    assert_near_equal(-42.0 / -14.0, DEFAULTS['h_V'], 1e-12)
    assert_near_equal(-126.0 / -42.0, DEFAULTS['h_V'], 1e-12)
    assert_near_equal(1470.0 / 490.0, DEFAULTS['h_V'], 1e-12)

    assert_near_equal(490.0 / 14.0, DEFAULTS['l_V'], 1e-12)
    assert_near_equal(1470.0 / 42.0, DEFAULTS['l_V'], 1e-12)
    assert_near_equal(17150.0 / 490.0, DEFAULTS['l_V'], 1e-12)


def test_interference_drag_amplifies_the_sideslip_row_by_seventeen_per_cent():
    """1/(1 - dD_int/Y_V_bar) = 1.175, not a rounding correction.

    The interference drag is itself proportional to the fin's side force, so a
    sideslip perturbation feeds back on itself. With dD_int set to zero the row
    loses both that factor and its direct term.
    """
    factor = 1.0 / (1.0 - DEFAULTS['D_int'] / DEFAULTS['Y_V_bar'])
    assert_near_equal(factor, 1.175, 5e-3)

    with_drag = run().get_val('dY_dydot')[0]
    without = run(D_int=0.0).get_val('dY_dydot')[0]
    assert abs(with_drag / without - 1.0) > 0.35


def test_interference_drag_is_three_quarters_of_the_fin_drag():
    """Chapter 8, p. 509: 42.8 lb against 14.9 lb of clean fin drag."""
    assert_near_equal(DEFAULTS['D_int'], 42.8, 1e-12)
    assert DEFAULTS['D_int'] > 2.5 * 14.9


def test_the_chapter_8_value_is_what_the_table_implies():
    """Inverting dY/dydot = -14 for dD_int gives 40.7 lb against 42.8."""
    S = (DEFAULTS['qV_q'] * DEFAULTS['q'] * DEFAULTS['A_V']
         * DEFAULTS['a_V'])
    tail = DEFAULTS['dY_dydot_T'] / DEFAULTS['T_T']
    lift = S * DEFAULTS['d_alphaV_d_ydot']
    inner = -2.0 / DEFAULTS['V'] + tail
    Y_V = DEFAULTS['Y_V_bar']

    # -14 (1 - D/Y_V) = lift + D inner  ->  D (14/Y_V - inner) = lift + 14
    recovered = (lift + 14.0) / (14.0 / Y_V - inner)
    assert_near_equal(recovered, 40.7, 3e-2)
    assert abs(recovered / DEFAULTS['D_int'] - 1.0) < 0.06


def test_dX_dydot_cannot_be_reached():
    """-4.21 against a printed -3, and the two constraints are incompatible.

    Reaching -3 needs the bracket K_V = -.070, hence alpha_V = -4.5 deg. But
    Y_V_bar = 287 lb over (q_V/q) q A_V a_V fixes alpha_V at +0.38 deg. Both
    cannot hold, and the row is a single digit in the book.
    """
    prob = run()
    assert_near_equal(prob.get_val('dX_dydot')[0], -4.21, 2e-2)

    S = (DEFAULTS['qV_q'] * DEFAULTS['q'] * DEFAULTS['A_V']
         * DEFAULTS['a_V'])
    from_force = DEFAULTS['Y_V_bar'] / S + DEFAULTS['alpha_LO']
    assert_near_equal(np.degrees(from_force), 0.38, 5e-2)
    assert_near_equal(from_force, DEFAULTS['alpha_V_bar'], 3e-3)


def test_dR_dxdot_contradicts_dN_dxdot():
    """The book's own two rows imply different values of dY/dxdot.

    dR/dxdot = (dY/dxdot) h_V = 12 needs dY/dxdot = 4.0, while
    dN/dxdot = -(dY/dxdot) l_V = -161 needs 4.6, with h_V = 3 and l_V = 35
    each confirmed three ways elsewhere in the table. The model gives 4.58,
    which reproduces dN/dxdot to 0.4 % and overshoots dR/dxdot by 15 %.
    """
    prob = run()
    assert_near_equal(12.0 / DEFAULTS['h_V'], 4.0, 1e-12)
    assert_near_equal(161.0 / DEFAULTS['l_V'], 4.6, 1e-2)

    got = prob.get_val('dY_dxdot')[0]
    assert abs(got / 4.6 - 1.0) < 0.01
    assert abs(prob.get_val('dN_dxdot')[0] / -161.0 - 1.0) < 0.01
    assert abs(prob.get_val('dR_dxdot')[0] / 12.0 - 1.0) > 0.10


def test_everything_driven_by_the_sideslip_row_shares_one_offset():
    """Nine rows sit 1.7 % high together, which is one number not nine."""
    prob = run()
    for name in ('dY_dydot', 'dY_dp', 'dY_dr', 'dR_dydot', 'dR_dp', 'dR_dr',
                 'dN_dydot', 'dN_dp', 'dN_dr'):
        ratio = abs(prob.get_val(name)[0] / TABLE_9_13[name])
        assert 1.015 < ratio < 1.020, name


def test_a_clean_fin_loses_the_feedback():
    """With no interference drag, dY/dydot is the plain lift slope."""
    prob = run(D_int=0.0)
    S = (DEFAULTS['qV_q'] * DEFAULTS['q'] * DEFAULTS['A_V']
         * DEFAULTS['a_V'])
    assert_near_equal(prob.get_val('dY_dydot')[0],
                      S * DEFAULTS['d_alphaV_d_ydot'], 1e-12)


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_13:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-12)
