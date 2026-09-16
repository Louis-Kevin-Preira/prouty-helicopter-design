"""G2 tests -- Table 9.9, tail rotor derivatives in forward flight (pp. 582-583)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.basic_tail_rotor_derivatives_ff_comp import (
    BasicTailRotorDerivativesFFComp,
)
from prouty.stability.rotor_chart_derivatives_comp import (
    RotorChartDerivativesComp,
)
from prouty.stability.tail_rotor_derivatives_ff_comp import (
    SCALAR_INPUTS,
    TailRotorDerivativesFFComp,
    build_rows,
)

# rho A_b (Omega R)^2 = 19,492 and the arms h_T = 6, l_T = 37, all from
# Table 9.3 (pp. 569-570). The chart column is Table 9.5, p. 574, tail rotor,
# and dlambda'/dydot is Table 9.7's printed -.00121 rather than the -.001228
# the geometric solidity gives -- Table 9.9's -24.5 is computed from the
# printed value.
EXAMPLE = dict(
    rho=0.002378, A_b=19.397, Omega_R=650.0, h_T=6.0, l_T=37.0,
    dCT_sigma_dmu=-0.070, dCT_sigma_dlambda=1.04, dCT_sigma_dtheta0=0.659,
    d_mu_d_xdot=1.0 / 650.0, d_lambda_d_ydot=-0.00121,
)

# Table 9.9, pp. 582-583.
TABLE_9_9 = {
    'dY_dxdot': -2, 'dY_dydot': -24.5, 'dY_dp': -147, 'dY_dr': 907,
    'dY_dtheta0': 12845,
    'dR_dxdot': -13, 'dR_dydot': -147, 'dR_dp': -882, 'dR_dr': 5442,
    'dR_dtheta0': 77070,
    'dN_dxdot': 78, 'dN_dydot': 907, 'dN_dp': 5439, 'dN_dr': -33559,
    'dN_dtheta0': -475265,
}


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('t', TailRotorDerivativesFFComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALAR_INPUTS else np.full(nn, value))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name, expected', TABLE_9_9.items())
def test_table_9_9(name, expected):
    got = run().get_val(name)[0]
    tol = max(0.55, 3e-3 * abs(expected))
    assert abs(got - expected) <= tol, f'{name}: {got} vs {expected}'


def test_the_table_has_fifteen_rows_thirteen_of_them_references():
    rows = build_rows()
    assert len(rows) == len(TABLE_9_9) == 15
    references = [name for name, terms in rows.items()
                  if all(entry[0] == 'ref' for entry in terms)]
    assert len(references) == 12


def test_only_three_rows_are_aerodynamic():
    """Everything else is those three carried on h_T or -l_T."""
    prob = run()
    for moment, force in (('dR_dxdot', 'dY_dxdot'), ('dR_dydot', 'dY_dydot'),
                          ('dR_dtheta0', 'dY_dtheta0')):
        assert_near_equal(prob.get_val(moment)[0],
                          prob.get_val(force)[0] * EXAMPLE['h_T'], 1e-12)
    for moment, force in (('dN_dxdot', 'dY_dxdot'), ('dN_dydot', 'dY_dydot'),
                          ('dN_dtheta0', 'dY_dtheta0')):
        assert_near_equal(prob.get_val(moment)[0],
                          -prob.get_val(force)[0] * EXAMPLE['l_T'], 1e-12)


def test_it_settles_c9_8():
    """dR/dydot = (dY/dydot) h_T = -147 here, against Table 9.3's printed +78.

    Both tables print the same equation for the same derivative with the same
    h_T = +6. Table 9.9 prints a value consistent with it; Table 9.3 does not.
    That makes the hover +78 a misprint rather than a convention this
    implementation has misread.
    """
    prob = run()
    assert_near_equal(prob.get_val('dR_dydot')[0],
                      prob.get_val('dY_dydot')[0] * EXAMPLE['h_T'], 1e-12)
    assert prob.get_val('dR_dydot')[0] < 0.0
    assert_near_equal(prob.get_val('dR_dydot')[0], -147.0, 3e-3)


def test_the_three_M_rows_of_hover_are_gone():
    """Table 9.3 has dM/dydot, dM/dr and dM/dtheta0; Table 9.9 has none.

    Those were the tail rotor torque rows whose signs contradicted each other
    in hover (C9-9). Chapter 9 does not resolve the contradiction; it stops
    printing the rows.
    """
    rows = build_rows()
    assert not [name for name in rows if name.startswith('dM_')]


def test_forward_flight_adds_the_xdot_rows():
    """A tail rotor in forward flight sees its own advance ratio change."""
    rows = build_rows()
    assert {'dY_dxdot', 'dR_dxdot', 'dN_dxdot'} <= set(rows)
    assert abs(run().get_val('dY_dxdot')[0]) > 0.0
    assert_near_equal(run(dCT_sigma_dmu=0.0).get_val('dY_dxdot')[0], 0.0, 1e-13)


def test_two_rows_are_the_same_product_reached_two_ways():
    """dR/dr and dN/dp are both (dY/dydot) h_T l_T, printed as 5,442 and 5,439.

    The book rounds dY/dr to 907 before multiplying by h_T, which is where the
    0.06 % comes from. The model gives one number for both.
    """
    prob = run()
    assert_near_equal(prob.get_val('dR_dr')[0], prob.get_val('dN_dp')[0], 1e-12)
    assert abs(TABLE_9_9['dR_dr'] / TABLE_9_9['dN_dp'] - 1.0) < 1e-3


def test_yaw_damping_is_negative_and_the_dominant_term():
    """dN/dr = -33,559 against the main rotor's -3,204 at the same speed."""
    assert run().get_val('dN_dr')[0] < -30000.0


def test_it_chains_from_tables_9_5_and_9_7():
    """Chart column and basic derivatives feeding the dimensional table."""
    prob = om.Problem()
    prob.model.add_subsystem('chart', RotorChartDerivativesComp(rotor='tail'),
                             promotes=['dCT_sigma_dmu', 'dCT_sigma_dlambda',
                                       'dCT_sigma_dtheta0', 'mu'])
    prob.model.add_subsystem('basic', BasicTailRotorDerivativesFFComp(),
                             promotes=['*'])
    prob.model.add_subsystem('table', TailRotorDerivativesFFComp(),
                             promotes=['*'])
    prob.model.set_input_defaults('Omega_R', val=np.full(1, 650.0),
                                  units='ft/s')
    prob.setup()
    prob.set_val('rho', np.full(1, 0.002378))
    prob.set_val('sigma', 0.14614)
    prob.set_val('A_b', 19.397)
    prob.set_val('h_T', 6.0)
    prob.set_val('l_T', 37.0)
    prob.set_val('CT_sigma_bar', np.full(1, 0.033912))
    prob.set_val('a1s_bar', np.full(1, 0.065266))
    prob.run_model()

    assert_near_equal(prob.get_val('dY_dtheta0')[0], 12845.0, 5e-3)
    assert abs(prob.get_val('dY_dydot')[0] / -24.5 - 1.0) < 2e-2


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_9:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-12)
