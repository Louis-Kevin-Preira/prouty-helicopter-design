"""G3 tests -- Tables 9.14 and 9.15, fuselage derivatives (pp. 589-591)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.fuselage_derivatives_ff_comp import (
    INPUT_DEFAULTS,
    SCALAR_INPUTS,
    FuselageDerivativesFFComp,
    build_rows,
)
from prouty.stability.fuselage_nondim_derivatives_comp import (
    APPENDIX_A,
    FuselageNondimDerivativesComp,
)
from prouty.stability.horiz_stab_nondim_derivatives_comp import (
    HorizStabNondimDerivativesComp,
)

# Table 9.14 inputs. v_F/v_1 = 1 is stated in Chapter 8's Table 8.4 and is
# already the value this repository's RotorDownwashComp uses.
NONDIM = dict(V=194.1, q=44.793, A_M=2827.4, vF_v1=1.0, Z_M_bar=-20543.0,
              dZ_dxdot_M=45.0, dZ_dzdot_M=-261.0)
NONDIM_SCALARS = ('A_M',)

TABLE_9_14 = {
    'd_gammaC_d_zdot': -0.00515, 'd_beta_d_ydot': 0.00515,
    'd_epsMF_d_xdot': -0.00051, 'd_epsMF_d_zdot': 0.00051,
    'd_alphaF_d_xdot': 0.00051, 'd_alphaF_d_zdot': 0.00467,
}

TABLE_9_15 = {
    'dX_dxdot': -8.0, 'dX_dzdot': -1.0, 'dY_dydot': -55.0, 'dZ_dxdot': 3.0,
    'dZ_dzdot': -19.0, 'dR_dydot': 53.0, 'dM_dxdot': -80.0, 'dM_dzdot': 374.0,
    'dN_dydot': -190.0,
}
#: Rows the book prints as a single digit, checked by rounding instead.
SINGLE_DIGIT = ('dX_dzdot', 'dZ_dxdot')


def run_nondim(nn=1, **overrides):
    values = dict(NONDIM, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('f', FuselageNondimDerivativesComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in NONDIM_SCALARS
                     else np.full(nn, value))
    prob.run_model()
    return prob


def run(nn=1, **overrides):
    values = dict(INPUT_DEFAULTS, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('f', FuselageDerivativesFFComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALAR_INPUTS
                     else np.full(nn, value))
    prob.run_model()
    return prob


# ------------------------------------------------------- Table 9.14, p. 589

@pytest.mark.parametrize('name, expected', TABLE_9_14.items())
def test_table_9_14(name, expected):
    got = run_nondim().get_val(name)[0]
    # two printed figures, so half a digit is the floor
    assert abs(got - expected) <= max(5e-6, 2e-2 * abs(expected)), \
        f'{name}: {got} vs {expected}'


def test_the_fuselage_sits_in_the_undeveloped_wake():
    """v_F/v_1 = 1 where the horizontal stabilizer has 1.5.

    Same 4 q A_M, same rotor Z derivative, so the two downwash rows differ by
    exactly the ratio: .00051 against .00077.
    """
    fuselage = run_nondim().get_val('d_epsMF_d_zdot')[0]

    stab = om.Problem()
    stab.model.add_subsystem('h', HorizStabNondimDerivativesComp(),
                             promotes=['*'])
    stab.setup()
    stab.set_val('vH_v1', np.full(1, 1.4946))
    stab.set_val('Z_M_bar', np.full(1, NONDIM['Z_M_bar']))
    stab.run_model()

    assert_near_equal(stab.get_val('d_epsMH_d_zdot')[0] / fuselage, 1.4946,
                      1e-6)


def test_there_is_no_upwash_row_and_no_lag_row():
    """The fuselage has no eps_F on itself and sits under the rotor, not aft."""
    prob = run_nondim()
    assert_near_equal(
        prob.get_val('d_alphaF_d_zdot')[0],
        -(prob.get_val('d_epsMF_d_zdot')[0]
          + prob.get_val('d_gammaC_d_zdot')[0]), 1e-13)

    names = set(TABLE_9_14)
    assert not any('zddot' in name or 'epsF' in name for name in names)


def test_the_six_appendix_a_rows_are_data_not_equations():
    """Table 9.14 lists them; they go straight into Table 9.15."""
    assert set(APPENDIX_A) <= set(INPUT_DEFAULTS)
    for name, value in APPENDIX_A.items():
        assert_near_equal(INPUT_DEFAULTS[name], value, 1e-13)


def test_nondim_partials():
    prob = run_nondim(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-9, rtol=1e-9)


# ------------------------------------------------------- Table 9.15, p. 590

@pytest.mark.parametrize('name, expected', TABLE_9_15.items())
def test_table_9_15(name, expected):
    got = run().get_val(name)[0]
    if name in SINGLE_DIGIT:
        assert round(got) == expected, f'{name}: {got} vs {expected}'
    else:
        assert abs(got / expected - 1.0) <= 3e-2, f'{name}: {got} vs {expected}'


def test_the_table_has_nine_rows():
    assert len(build_rows()) == len(TABLE_9_15) == 9


def test_the_dynamic_pressure_rows_need_no_curve():
    """dX/dxdot, dZ/dxdot and half of dM/dxdot are (2/V) times a trim force."""
    prob = run()
    assert_near_equal(prob.get_val('dX_dxdot')[0],
                      2.0 * INPUT_DEFAULTS['X_F_bar'] / INPUT_DEFAULTS['V'],
                      1e-12)
    assert_near_equal(prob.get_val('dZ_dxdot')[0],
                      2.0 * INPUT_DEFAULTS['Z_F_bar'] / INPUT_DEFAULTS['V'],
                      1e-12)

    curve_free = run(dMq_dalphaF=0.0).get_val('dM_dxdot')[0]
    assert_near_equal(curve_free,
                      2.0 * INPUT_DEFAULTS['M_F_bar'] / INPUT_DEFAULTS['V'],
                      1e-12)


def test_chordwise_and_normal_force_use_different_curves():
    """dX/dzdot follows df/dalpha_F, dZ/dzdot follows d(L/q)/dalpha_F."""
    base = run()
    no_drag_area = run(df_dalphaF=0.0)
    no_lift_area = run(dLq_dalphaF=0.0)

    assert not np.isclose(base.get_val('dX_dzdot')[0],
                          no_drag_area.get_val('dX_dzdot')[0])
    assert np.isclose(base.get_val('dX_dzdot')[0],
                      no_lift_area.get_val('dX_dzdot')[0])
    assert not np.isclose(base.get_val('dZ_dzdot')[0],
                          no_lift_area.get_val('dZ_dzdot')[0])
    assert np.isclose(base.get_val('dZ_dzdot')[0],
                      no_drag_area.get_val('dZ_dzdot')[0])


def test_the_trim_forces_are_the_chapter_8_ones():
    """L_F_bar = -281 and D_F_bar = 794, Table 8.5 p. 524, unadjusted."""
    assert_near_equal(INPUT_DEFAULTS['L_F_bar'], -281.0, 1e-13)
    assert_near_equal(INPUT_DEFAULTS['D_F_bar'], 794.0, 1e-13)
    assert_near_equal(INPUT_DEFAULTS['X_F_bar'], -INPUT_DEFAULTS['D_F_bar'],
                      1e-13)
    assert_near_equal(INPUT_DEFAULTS['Z_F_bar'], -INPUT_DEFAULTS['L_F_bar'],
                      1e-13)


def test_the_fuselage_is_the_bigger_half_of_the_sideslip_damping():
    """dY/dydot = -55 against the vertical stabilizer's -14 (Table 9.13)."""
    assert run().get_val('dY_dydot')[0] < 3.0 * -14.0


def test_yaw_and_roll_from_sideslip_have_opposite_signs():
    """d(N/q)/dbeta = -820 stabilizing, d(R/q)/dbeta = +230."""
    prob = run()
    assert prob.get_val('dN_dydot')[0] < 0.0
    assert prob.get_val('dR_dydot')[0] > 0.0


def test_M_F_bar_is_what_dM_dxdot_implies():
    """-11,733 ft-lb, the one trim value Chapter 8 does not supply."""
    curve = (INPUT_DEFAULTS['q'] * INPUT_DEFAULTS['dMq_dalphaF']
             * INPUT_DEFAULTS['d_alphaF_d_xdot'])
    recovered = (-80.0 - curve) * INPUT_DEFAULTS['V'] / 2.0
    assert_near_equal(recovered, INPUT_DEFAULTS['M_F_bar'], 1e-2)


def test_it_chains_from_table_9_14():
    prob = om.Problem()
    prob.model.add_subsystem('nondim', FuselageNondimDerivativesComp(),
                             promotes=['*'])
    prob.model.add_subsystem('dim', FuselageDerivativesFFComp(), promotes=['*'])
    prob.model.set_input_defaults('V', val=np.full(1, 194.1), units='ft/s')
    prob.model.set_input_defaults('q', val=np.full(1, 44.793),
                                  units='lbf/ft**2')
    prob.setup()
    prob.set_val('Z_M_bar', np.full(1, NONDIM['Z_M_bar']))
    prob.run_model()

    assert abs(prob.get_val('dZ_dzdot')[0] / -19.0 - 1.0) < 3e-2
    assert abs(prob.get_val('dM_dzdot')[0] / 374.0 - 1.0) < 3e-2


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_15:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-12)
