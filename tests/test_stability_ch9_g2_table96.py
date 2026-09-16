"""G2 tests -- Table 9.6, basic main rotor derivatives in forward flight."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability.basic_main_rotor_derivatives_ff_comp import (
    SCALAR_INPUTS,
    BasicMainRotorDerivativesFFComp,
)
from prouty.stability.basic_rotor_derivatives_hover_comp import (
    BasicRotorDerivativesHoverComp,
)

# Example helicopter, level flight at 115 knots (mu = .30). The trim inflow
# and tip-path-plane angle are not printed in the section; lambda' = -.023 is
# the Table 9.5 trim condition (p. 574) and alpha_TPP is recovered below from
# dlambda'/dxdot = .000037.
EXAMPLE = dict(
    e_over_R=0.05, a=6.0, sigma=0.085, R=30.0, A_b=240.0,
    rho=0.002378, Omega=650.0 / 30.0, Omega_R=650.0, V=194.1,
    gamma=8.1, mu=0.30, CT_sigma_bar=0.0865, lambda_bar=-0.023,
    alpha_TPP_bar=-0.036631,
    dCT_sigma_dmu=-0.140, dCT_sigma_dlambda=0.79,
)

# Table 9.6, pp. 576-577, example helicopter column.
TABLE_9_6 = {
    'd_mu_d_xdot': 0.00154,
    'd_lambda_d_xdot': 0.000037,
    'd_lambda_d_zdot': 0.00138,
    'd_beta_d_ydot': 0.00515,
    'dCH_sigma_da1s': 0.069,
    'd_a1s_dq': -0.1098,
    'd_a1s_dp': 0.0396,
    'd_a1s_dA1': 0.0905,
    'd_a1s_dB1': -1.188,
    'd_b1s_dq': -0.0354,
    'd_b1s_dp': -0.101,
    'd_b1s_dA1': 1.0,
    'd_b1s_dB1': 0.0983,
    'dM_da1s': 200940.0,
}


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('basic', BasicMainRotorDerivativesFFComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALAR_INPUTS else np.full(nn, value))
    prob.run_model()
    return prob


@pytest.mark.parametrize('name, expected', TABLE_9_6.items())
def test_table_9_6(name, expected):
    """Every printed value, to the precision it is printed with."""
    got = run().get_val(name)[0]
    tol = max(5e-7, 5e-3 * abs(expected))
    assert abs(got - expected) <= tol, f'{name}: {got} vs {expected}'


def test_mirrored_rows():
    """Table 9.6 pairs dCH/da1s with dCY/db1s, and dM/da1s with dR/db1s."""
    prob = run()
    assert_near_equal(prob.get_val('dCY_sigma_db1s')[0],
                      prob.get_val('dCH_sigma_da1s')[0], 1e-13)
    assert_near_equal(prob.get_val('dR_db1s')[0],
                      prob.get_val('dM_da1s')[0], 1e-13)


def test_inflow_derivative_with_forward_speed():
    """dlambda'/dxdot is two orders below its hover counterpart.

    In hover a fore-and-aft velocity does not change the inflow at all beyond
    the disc tilt; at mu = .30 it is .000037 against dmu/dxdot = .00154, a
    factor of forty smaller, because the induced part now falls as 1/mu.
    """
    prob = run()
    assert 0.0 < prob.get_val('d_lambda_d_xdot')[0] < 0.05 * prob.get_val(
        'd_mu_d_xdot')[0]


def test_alpha_tpp_is_what_the_printed_value_implies():
    """Solving dlambda'/dxdot = .000037 backwards gives about -2.1 degrees."""
    sigma, mu, x_T = EXAMPLE['sigma'], EXAMPLE['mu'], EXAMPLE['CT_sigma_bar']
    induced = 0.5 * sigma / mu * (EXAMPLE['dCT_sigma_dmu'] - x_T / mu)
    alpha = 0.000037 * EXAMPLE['Omega_R'] + induced
    assert_near_equal(np.degrees(alpha), -2.1, 5e-2)


def test_heave_inflow_is_damped_by_the_thrust_response():
    """dlambda'/dzdot = .00138 against 1/(Omega R) = .00154.

    The rotor's own thrust change absorbs part of the inflow perturbation, the
    same feedback the hover version carries in a different form.
    """
    prob = run()
    assert prob.get_val('d_lambda_d_zdot')[0] < prob.get_val('d_mu_d_xdot')[0]
    assert_near_equal(prob.get_val('d_lambda_d_zdot')[0]
                      / prob.get_val('d_mu_d_xdot')[0], 1.0 / 1.1119, 1e-3)


def test_six_of_eight_flapping_rows_collapse_onto_table_9_1():
    """At mu = 0 the forward-flight forms must reduce to the hover ones."""
    ff = run(mu=1e-9)
    hover = om.Problem()
    hover.model.add_subsystem('h', BasicRotorDerivativesHoverComp(),
                              promotes=['*'])
    hover.setup()
    for name in ('e_over_R', 'a', 'sigma', 'R', 'A_b'):
        hover.set_val(name, EXAMPLE[name])
    for name in ('rho', 'Omega', 'Omega_R', 'gamma'):
        hover.set_val(name, np.full(1, EXAMPLE[name]))
    hover.run_model()

    for name in ('d_a1s_dq', 'd_a1s_dp', 'd_a1s_dA1', 'd_b1s_dq', 'd_b1s_dp',
                 'd_b1s_dB1', 'dM_da1s'):
        assert_near_equal(ff.get_val(name)[0], hover.get_val(name)[0], 1e-6)


def test_the_two_cyclic_rows_that_disagree_with_table_9_1():
    """C9-13: Table 9.6 drops the flapping determinant, Table 9.1 keeps it."""
    ff = run(mu=1e-9)
    kappa = ff.get_val('d_a1s_dA1')[0]

    assert_near_equal(ff.get_val('d_a1s_dB1')[0], -1.0, 1e-6)
    assert_near_equal(ff.get_val('d_b1s_dA1')[0], 1.0, 1e-13)
    # Table 9.1 would give -1/(1+kappa^2) and +1/(1+kappa^2)
    assert abs(1.0 / (1.0 + kappa ** 2) - 1.0) < 0.01
    assert abs(1.0 / (1.0 + kappa ** 2) - 1.0) > 1e-3


def test_advance_ratio_moves_the_cyclic_response():
    """da1s/dB1 goes from -1 in hover to -1.188 at mu = .30."""
    assert_near_equal(run(mu=1e-9).get_val('d_a1s_dB1')[0], -1.0, 1e-6)
    assert_near_equal(run().get_val('d_a1s_dB1')[0], -1.188, 5e-4)


def test_db1s_dA1_is_exactly_one_at_every_speed():
    """The one row Table 9.6 prints as a bare 1."""
    for mu in (0.05, 0.15, 0.30, 0.45):
        assert_near_equal(run(mu=mu).get_val('d_b1s_dA1')[0], 1.0, 1e-13)


def test_the_inflow_rows_diverge_as_mu_goes_to_zero():
    """Table 9.6 is a forward-flight table and cannot be run down to hover.

    Both lambda' rows carry sigma/(2 mu) and the thrust term carries
    (C_T/sigma)/mu, from the Glauert high-speed induced velocity
    v_1/(Omega R) = C_T/(2 mu). As mu falls, dlambda'/dxdot grows without bound
    and dlambda'/dzdot collapses to zero. This is why Chapter 9 keeps Tables
    9.1 to 9.4 for hover and 9.5 to 9.9 for forward flight, and why the two
    are separate groups here.
    """
    slow, crawl, fast = run(mu=0.05), run(mu=0.005), run(mu=0.30)

    # dlambda'/dxdot blows up: 65 times larger at mu = .05 than at mu = .30
    assert (abs(slow.get_val('d_lambda_d_xdot')[0])
            > 20.0 * abs(fast.get_val('d_lambda_d_xdot')[0]))

    # dlambda'/dzdot goes the other way, falling off proportionally to mu
    assert (crawl.get_val('d_lambda_d_zdot')[0]
            < 0.25 * fast.get_val('d_lambda_d_zdot')[0])
    assert 1.6 < (run(mu=0.01).get_val('d_lambda_d_zdot')[0]
                  / crawl.get_val('d_lambda_d_zdot')[0]) < 2.0

    # the flapping rows, by contrast, are perfectly well behaved at mu = 0
    assert np.isfinite(run(mu=0.0).get_val('d_a1s_dq')[0])


def test_teetering_rotor_loses_the_hub_spring():
    prob = run(e_over_R=0.0)
    for name in ('dM_da1s', 'dR_db1s', 'd_a1s_dA1', 'd_b1s_dB1'):
        assert_near_equal(prob.get_val(name)[0], 0.0, 1e-13)


@pytest.mark.parametrize('e_over_R', (0.0, 0.05))
@pytest.mark.parametrize('mu', (0.15, 0.30))
def test_partials(e_over_R, mu):
    prob = run(nn=3, e_over_R=e_over_R, mu=mu)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in TABLE_9_6:
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-13)
