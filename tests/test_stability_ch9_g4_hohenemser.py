"""G4 tests -- the Hohenemser hover period (pp. 600-601)."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.stability import describe_modes
from prouty.stability.hohenemser_period_comp import HohenemserPeriodComp

EXAMPLE = dict(
    dM_dxdot=143.0, dM_dq=-28659.0,
    d_a1s_d_mu=0.34, d_a1s_dq=-0.105, Omega_R=650.0,
    CT_sigma_bar=0.086, gamma=8.1, g=32.2, a=6.0, R=30.0,
)
SCALARS = ('a', 'R')

BOOK_PERIOD = 15.7          # p. 600
BOOK_RULE = 3.2             # p. 601, P = 3.2 sqrt(R)


def run(nn=1, **overrides):
    values = dict(EXAMPLE, **overrides)
    prob = om.Problem()
    prob.model.add_subsystem('h', HohenemserPeriodComp(num_nodes=nn),
                             promotes=['*'])
    prob.setup(force_alloc_complex=True)
    for name, value in values.items():
        prob.set_val(name, value if name in SCALARS else np.full(nn, value))
    prob.run_model()
    return prob


def test_period_p600():
    """15.7 seconds from the derivative form."""
    assert_near_equal(run().get_val('period')[0], BOOK_PERIOD, 3e-3)


def test_flapping_form_p600():
    """The second printed form gives the same period."""
    prob = run()
    assert_near_equal(prob.get_val('period_flapping')[0], BOOK_PERIOD, 3e-3)
    assert abs(prob.get_val('period_flapping')[0]
               / prob.get_val('period')[0] - 1.0) < 3e-3


def test_the_flapping_form_is_exact_not_an_approximation():
    """The h_M terms cancel in the ratio, so the two forms are identical.

    Rebuilt from the Table 9.1 pieces rather than the rounded Table 9.4
    values, the derivative form and the flapping form agree to machine
    precision. Table 9.2 gives dM/dxdot and dM/dq the same
    (dM/da1s + K h_M) factor, and it cancels.
    """
    scale, dCH, h_M, dM_da1s = 241131.0, 0.0398, 7.5, 200940.0
    a_mu, a_q, mu_x = EXAMPLE['d_a1s_d_mu'], EXAMPLE['d_a1s_dq'], 1.0 / 650.0

    dX_dxdot = -scale * dCH * a_mu * mu_x
    dX_dq = -scale * dCH * a_q
    dM_dxdot = dM_da1s * a_mu * mu_x - dX_dxdot * h_M
    dM_dq = dM_da1s * a_q - dX_dq * h_M

    prob = run(dM_dxdot=dM_dxdot, dM_dq=dM_dq)
    assert_near_equal(prob.get_val('omega_N_squared')[0],
                      prob.get_val('omega_N_squared_flapping')[0], 1e-12)


def test_it_matches_the_two_dof_oscillation():
    """15.7 s against the 17.7 s of the full cubic: the inertia is worth 13 %."""
    from test_stability_ch9_g4_two_dof import run as run_two_dof

    hohenemser = run().get_val('period')[0]
    exact = [m for m in describe_modes(
        run_two_dof().get_val('char_coeffs')[0]) if m.oscillatory][0].period

    assert hohenemser < exact
    assert 0.85 < hohenemser / exact < 0.92


def test_radius_form_p601():
    """P = 2 pi sqrt(R)/sqrt(g (C_T/sigma) gamma/a), and its 3.2 sqrt(R) rule."""
    prob = run()
    coefficient = prob.get_val('period_radius')[0] / np.sqrt(EXAMPLE['R'])
    assert_near_equal(coefficient, BOOK_RULE, 2e-2)


def test_radius_form_scales_as_root_R():
    """Quadrupling the radius doubles the period."""
    assert_near_equal(run(R=120.0).get_val('period_radius')[0]
                      / run().get_val('period_radius')[0], 2.0, 1e-12)


def test_the_radius_form_rests_on_two_substitutions():
    """p. 601 replaces da1s/dmu by 16 (C_T/sigma)/a and da1s/dq by 16/(gamma Omega).

    The first is Prouty's "small white lie": Chapter 3 at mu = 0 carries
    4 v_1/(Omega R) where da1s/dmu carries 2, so the substitution drops
    2 v_1/(Omega R) and turns .34 into .229. The second sets the hinge offset
    to zero, which the example helicopter does not have. Each moves the period
    on its own; together they give period_radius.
    """
    exact = run().get_val('period_flapping')[0]
    lied = 16.0 * EXAMPLE['CT_sigma_bar'] / EXAMPLE['a']
    Omega = EXAMPLE['Omega_R'] / EXAMPLE['R']
    assert_near_equal(lied, 0.229, 2e-2)

    # the white lie alone stretches the period by about a fifth
    white_lie = run(d_a1s_d_mu=lied).get_val('period_flapping')[0]
    assert 1.18 < white_lie / exact < 1.25

    # both substitutions together reproduce the radius form
    both = run(d_a1s_d_mu=lied,
               d_a1s_dq=-16.0 / (EXAMPLE['gamma'] * Omega))
    assert_near_equal(both.get_val('period_flapping')[0],
                      both.get_val('period_radius')[0], 5e-3)
    assert 1.10 < both.get_val('period_radius')[0] / exact < 1.18


def test_damping_derivative_must_be_over_gamma_omega():
    """C9-12: p. 601 prints 16/(gamma R); only 16/(gamma Omega) gives sqrt(R).

    At a single radius the two are merely different numbers. The discriminator
    is the scaling. Holding tip speed fixed and varying R, so that
    Omega = (Omega R)/R:

        16/(gamma Omega) -> omega^2 = g (da1s/dmu) gamma / (16 R),  P ~ sqrt(R)
        16/(gamma R)     -> omega^2 = g (da1s/dmu) gamma / (16 Omega), P ~ 1/sqrt(R)

    The printed radius formula goes as sqrt(R), so the printed damping
    derivative cannot be the one that produced it -- and the two candidates
    move the period in opposite directions.
    """
    lied = 16.0 * EXAMPLE['CT_sigma_bar'] / EXAMPLE['a']
    tip_speed = EXAMPLE['Omega_R']

    def period(radius, over_omega):
        divisor = (EXAMPLE['gamma'] * tip_speed / radius if over_omega
                   else EXAMPLE['gamma'] * radius)
        return run(R=radius, d_a1s_d_mu=lied,
                   d_a1s_dq=-16.0 / divisor).get_val('period_flapping')[0]

    assert_near_equal(period(120.0, True) / period(30.0, True), 2.0, 1e-10)
    assert_near_equal(period(120.0, False) / period(30.0, False), 0.5, 1e-10)

    # and only the Omega version matches the printed radius formula
    assert_near_equal(period(30.0, True),
                      run().get_val('period_radius')[0], 5e-3)


def test_lateral_oscillation_shares_the_period():
    """p. 600: with no fuselage inertia the result applies to roll as well.

    Nothing about I_yy survives the reduction, so the same number serves both
    axes. This is the assertion that the component has no inertia input.
    """
    prob = om.Problem()
    prob.model.add_subsystem('h', HohenemserPeriodComp(), promotes=['*'])
    prob.setup()
    prob.final_setup()
    names = {m['prom_name'] for _, m
             in prob.model.list_inputs(out_stream=None, prom_name=True,
                                       val=False)}
    assert not any(n.startswith('I_') for n in names)


def test_a_divergent_case_returns_no_period():
    """Reversing dM/dxdot makes omega^2 negative: a divergence, not a mode."""
    with pytest.warns(UserWarning, match='divergence'):
        prob = run(dM_dxdot=-143.0)
    assert prob.get_val('omega_N_squared')[0] < 0.0
    assert_near_equal(prob.get_val('period')[0], 0.0, 1e-13)


def test_partials():
    prob = run(nn=3)
    data = prob.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_vectorized_matches_scalar():
    single, triple = run(), run(nn=3)
    for name in ('omega_N_squared', 'period', 'period_flapping',
                 'period_radius'):
        assert_near_equal(triple.get_val(name),
                          np.full(3, single.get_val(name)[0]), 1e-13)
