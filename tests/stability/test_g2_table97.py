"""G2 tests -- Table 9.7, basic tail rotor derivatives in forward flight."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.basic_main_rotor_derivatives_ff_comp import (
    BasicMainRotorDerivativesFFComp,
)
from prouty.stability.basic_tail_rotor_derivatives_ff_comp import (
    SCALAR_INPUTS,
    BasicTailRotorDerivativesFFComp,
)
from prouty.stability.rotor_chart_derivatives_comp import (
    RotorChartDerivativesComp,
)

# Tail rotor at 115 knots. sigma = A_b/(pi R^2) with A_b = 19.40 ft^2 and
# R = 6.5 ft, both from Table 9.3 (pp. 569-570). C_T/sigma = T_T/[rho A_b
# (Omega R)^2] with T_T = 661 lb, the Chapter 8 value (p. 510). a1s_bar is
# recovered from the printed dlambda'/dxdot -- see the test below.
EXAMPLE = dict(
    sigma=0.14614, Omega_R=650.0, V=194.1, mu=0.30,
    CT_sigma_bar=0.033912, a1s_bar=0.065266,
    dCT_sigma_dmu=-0.070, dCT_sigma_dlambda=1.04,
)

# Table 9.7, p. 578.
TABLE_9_7 = {'d_mu_d_xdot': 0.00154, 'd_lambda_d_xdot': 0.000169,
             'd_lambda_d_ydot': -0.00121, 'd_beta_d_ydot': 0.00515}

#: The one row the geometric solidity does not close. See the component.
ROUNDING = {'d_lambda_d_ydot': 2e-2}


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('basic', BasicTailRotorDerivativesFFComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALAR_INPUTS else np.full(nn, value))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name, expected', TABLE_9_7.items())
def test_table_9_7(name, expected):
    got = run().get_val(name)[0]
    tol = ROUNDING.get(name, 5e-3)
    assert abs(got / expected - 1.0) <= tol, f'{name}: {got} vs {expected}'


def test_the_inflow_row_is_negative():
    """p. 578 prints a leading minus, as Table 9.1 does in hover (C9-3)."""
    prob = run()
    assert prob.get_val('d_lambda_d_ydot')[0] < 0.0
    assert prob.get_val('d_mu_d_xdot')[0] > 0.0


def test_it_mirrors_the_main_rotor_row_it_pairs_with():
    """dlambda'/dydot on the tail is -dlambda'/dzdot on the main, same form.

    Fed the same solidity, advance ratio and chart slope, the two rows are
    equal and opposite. What differs in practice is only that the two rotors
    carry different numbers.
    """
    tail = run(sigma=0.085, dCT_sigma_dlambda=0.79)

    main = om.Problem()
    main.model.add_subsystem('m', BasicMainRotorDerivativesFFComp(),
                             promotes=['*'])
    main.setup()
    for name in ('sigma',):
        main.set_val(name, 0.085)
    for name, value in (('Omega_R', 650.0), ('mu', 0.30),
                        ('dCT_sigma_dlambda', 0.79)):
        main.set_val(name, np.full(1, value))
    main.run_model()

    assert_near_equal(tail.get_val('d_lambda_d_ydot')[0],
                      -main.get_val('d_lambda_d_zdot')[0], 1e-12)


def test_forward_speed_reaches_the_disc_only_through_flapping():
    """a1s_bar replaces alpha_TPP: at zero flapping only the induced term left."""
    prob = run(a1s_bar=0.0)
    induced = -0.5 * EXAMPLE['sigma'] / EXAMPLE['mu'] * (
        EXAMPLE['dCT_sigma_dmu'] - EXAMPLE['CT_sigma_bar'] / EXAMPLE['mu'])
    assert_near_equal(prob.get_val('d_lambda_d_xdot')[0],
                      induced / EXAMPLE['Omega_R'], 1e-12)


def test_a1s_bar_is_what_the_printed_value_implies():
    """Inverting dlambda'/dxdot = .000169 gives 3.74 degrees of flapping."""
    half = 0.5 * EXAMPLE['sigma'] / EXAMPLE['mu']
    induced = half * (EXAMPLE['dCT_sigma_dmu']
                      - EXAMPLE['CT_sigma_bar'] / EXAMPLE['mu'])
    a1s = 0.000169 * EXAMPLE['Omega_R'] + induced
    assert_near_equal(np.degrees(a1s), 3.74, 2e-2)
    assert_near_equal(a1s, EXAMPLE['a1s_bar'], 1e-3)


def test_the_row_that_does_not_close_needs_a_different_solidity():
    """dlambda'/dydot: -.001228 here against a printed -.00121.

    That row depends only on sigma_T, mu and dCT/sigma/dlambda'. Inverting the
    printed value would need sigma_T = .157, seven per cent above the .1461
    Table 9.3's A_b = 19.40 and R = 6.5 give. The geometry is taken as correct
    and the gap as rounding.
    """
    printed, mu, D_lam = -0.00121, EXAMPLE['mu'], EXAMPLE['dCT_sigma_dlambda']
    required = 2.0 * mu * (-1.0 / (printed * EXAMPLE['Omega_R']) - 1.0) / D_lam

    assert_near_equal(required, 0.157, 2e-2)
    assert required / EXAMPLE['sigma'] > 1.05

    assert_near_equal(run(sigma=required).get_val('d_lambda_d_ydot')[0],
                      printed, 1e-3)


def test_sideslip_row_is_shared_with_the_main_rotor():
    """dbeta/dydot = 1/V, the helicopter's airspeed, so both tables print .00515."""
    assert_near_equal(run().get_val('d_beta_d_ydot')[0],
                      1.0 / EXAMPLE['V'], 1e-13)


def test_it_takes_the_tail_column_of_table_9_5():
    """Wired to the chart component with rotor='tail'."""
    prob = om.Problem()
    prob.model.add_subsystem('chart', RotorChartDerivativesComp(rotor='tail'),
                             promotes=['dCT_sigma_dmu', 'dCT_sigma_dlambda',
                                       'mu'])
    prob.model.add_subsystem('basic', BasicTailRotorDerivativesFFComp(),
                             promotes=['*'])
    prob.setup()
    prob.set_val('sigma', EXAMPLE['sigma'])
    prob.set_val('CT_sigma_bar', np.full(1, EXAMPLE['CT_sigma_bar']))
    prob.set_val('a1s_bar', np.full(1, EXAMPLE['a1s_bar']))
    prob.run_model()

    assert_near_equal(prob.get_val('dCT_sigma_dmu')[0], -0.070, 1e-12)
    assert_near_equal(prob.get_val('d_lambda_d_xdot')[0], 0.000169, 5e-3)


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_7:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-13)
