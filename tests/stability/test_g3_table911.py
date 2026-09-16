"""G3 tests -- Table 9.11, horizontal stabilizer derivatives in forward flight."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.horiz_stab_derivatives_ff_comp import (
    INPUT_DEFAULTS,
    SCALAR_INPUTS,
    HorizStabDerivativesFFComp,
    build_rows,
)
from prouty.stability.horiz_stab_nondim_derivatives_comp import (
    HorizStabNondimDerivativesComp,
)

# Table 9.11, pp. 585-586. Rows the book prints as "<1" are given the model
# value and checked only for magnitude; see test_the_small_rows.
TABLE_9_11 = {
    'dX_dzdot': -1.0, 'dZ_dxdot': 1.0, 'dZ_dzdot': -7.0, 'dZ_dq': -217.0,
    'dM_dxdot': 42.0, 'dM_dzdot': -219.0, 'dM_dzddot': 9.0, 'dM_dq': -7161.0,
}
SMALL = ('dX_dxdot', 'dX_dzddot', 'dZ_dzddot')


def run(nn=1, **overrides):
    values = dict(INPUT_DEFAULTS, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('h', HorizStabDerivativesFFComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALAR_INPUTS else np.full(nn, value))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name, expected', TABLE_9_11.items())
def test_table_9_11(name, expected):
    got = run().get_val(name)[0]
    assert abs(got - expected) <= max(0.55, 1e-2 * abs(expected)), \
        f'{name}: {got} vs {expected}'


def test_the_table_has_eleven_rows():
    assert len(build_rows()) == len(TABLE_9_11) + len(SMALL) == 11


@pytest.mark.parametrize('name', SMALL)
def test_the_small_rows(name):
    """Three rows the book prints only as "<1" or "<1 lb/ft/sec^2"."""
    assert abs(run().get_val(name)[0]) < 1.0


def test_the_two_brackets_do_all_the_work():
    """Every row is one of two aerodynamic slopes carried onto an arm."""
    prob = run()
    assert_near_equal(prob.get_val('dZ_dq')[0],
                      prob.get_val('dZ_dzdot')[0] * INPUT_DEFAULTS['l_H'], 1e-12)
    assert_near_equal(prob.get_val('dM_dq')[0],
                      prob.get_val('dZ_dq')[0] * INPUT_DEFAULTS['l_H'], 1e-12)
    assert_near_equal(
        prob.get_val('dM_dzdot')[0],
        -prob.get_val('dX_dzdot')[0] * INPUT_DEFAULTS['h_H']
        + prob.get_val('dZ_dzdot')[0] * INPUT_DEFAULTS['l_H'], 1e-12)


def test_the_dynamic_pressure_term_dominates_dZ_dxdot():
    """(2/V) Z_H_bar is 2.79 of the 1.29 total, the rest coming back as -1.50."""
    prob = run()
    pressure = 2.0 * INPUT_DEFAULTS['Z_H_bar'] / INPUT_DEFAULTS['V']
    assert_near_equal(pressure, 2.79, 2e-2)
    assert pressure > 2.0 * prob.get_val('dZ_dxdot')[0]

    no_pressure = run(Z_H_bar=0.0).get_val('dZ_dxdot')[0]
    assert_near_equal(prob.get_val('dZ_dxdot')[0] - no_pressure, pressure, 1e-10)


def test_the_lag_rows_are_the_zdot_rows_scaled():
    """dX/dzddot and dZ/dzddot carry the Table 9.10 downwash lag ratio."""
    prob = run()
    ratio = (INPUT_DEFAULTS['d_alphaH_d_zddot']
             / INPUT_DEFAULTS['d_alphaH_d_zdot'])
    assert_near_equal(ratio, -0.040, 2e-2)

    for lag, steady in (('dX_dzddot', 'dX_dzdot'), ('dZ_dzddot', 'dZ_dzdot')):
        assert_near_equal(prob.get_val(lag)[0],
                          prob.get_val(steady)[0] * ratio, 1e-12)


def test_dM_dzddot_is_the_nine_s_squared_of_table_9_20():
    """Table 9.20's M row carries 9 s^2 (p. 614); this row is where it comes from."""
    assert_near_equal(run().get_val('dM_dzddot')[0], 9.0, 4e-2)


def test_no_lag_no_zddot_rows():
    assert_near_equal(run(d_alphaH_d_zddot=0.0).get_val('dM_dzddot')[0], 0.0,
                      1e-13)


def test_the_two_brackets_have_opposite_character():
    """Lift bracket goes with angle of attack, drag bracket is near unity."""
    prob = run()
    scale = (INPUT_DEFAULTS['qH_q'] * INPUT_DEFAULTS['q']
             * INPUT_DEFAULTS['A_H'] * INPUT_DEFAULTS['a_H'])
    lift = prob.get_val('dX_dzdot')[0] / (scale
                                          * INPUT_DEFAULTS['d_alphaH_d_zdot'])
    drag = -prob.get_val('dZ_dzdot')[0] / (scale
                                           * INPUT_DEFAULTS['d_alphaH_d_zdot'])
    assert_near_equal(lift, -0.1473, 1e-3)
    assert_near_equal(drag, 1.0192, 1e-3)


def test_a_symmetric_stabilizer_at_zero_angle():
    """At alpha_H = alpha_LO = i_H = 0 the lift bracket vanishes and drag is
    1 + C_D0, so dX/dzdot goes to zero and dZ/dzdot is pure area."""
    prob = run(alpha_H_bar=0.0, alpha_LO=0.0, i_H=0.0)
    assert_near_equal(prob.get_val('dX_dzdot')[0], 0.0, 1e-13)

    scale = (INPUT_DEFAULTS['qH_q'] * INPUT_DEFAULTS['q']
             * INPUT_DEFAULTS['A_H'] * INPUT_DEFAULTS['a_H'])
    assert_near_equal(prob.get_val('dZ_dzdot')[0],
                      -scale * (1.0 + INPUT_DEFAULTS['C_D0'])
                      * INPUT_DEFAULTS['d_alphaH_d_zdot'], 1e-12)


def test_it_chains_from_table_9_10():
    """Nondimensional derivatives in, dimensional ones out."""
    prob = om.Problem()
    prob.model.add_subsystem('nondim', HorizStabNondimDerivativesComp(),
                             promotes=['*'])
    prob.model.add_subsystem('dim', HorizStabDerivativesFFComp(),
                             promotes=['*'])
    prob.model.set_input_defaults('V', val=np.full(1, 194.1), units='ft/s')
    prob.model.set_input_defaults('l_H', val=33.12, units='ft')
    prob.setup()
    prob.set_val('vH_v1', np.full(1, 1.4946))
    prob.set_val('Z_M_bar', np.full(1, -20543.0))
    prob.set_val('deps_F_dalpha_F', np.full(1, 0.23293))
    prob.run_model()

    assert abs(prob.get_val('dM_dq')[0] / -7161.0 - 1.0) < 2e-2
    assert abs(prob.get_val('dZ_dzdot')[0] / -6.5 - 1.0) < 5e-2


def test_partials():
    prob = run(nn=3, alpha_LO=-0.03)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in build_rows():
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-12)
