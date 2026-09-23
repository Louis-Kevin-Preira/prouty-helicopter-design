"""Chapter 7 -- flapping due to pitch and roll velocities, p. 469-476."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.units import convert_units

from prouty.flapping import RateFlappingGroup
from prouty.flapping import flapping_2x2 as f2
from prouty.flapping.flapping_superposition_comp import FlappingSuperpositionComp
from prouty.flapping.lateral_cyclic_turn_comp import LateralCyclicTurnComp
from prouty.flapping.longitudinal_cyclic_turn_comp import \
    LongitudinalCyclicTurnComp
from prouty.flapping.maneuver_rate_comp import ManeuverRateComp
from prouty.flapping.rate_flapping_comp import RateFlappingComp


# ------------------------------------------------------------------------
# rate_flapping_comp
# ------------------------------------------------------------------------

GAMMA_EX_rate_flapping, E_OVER_R_EX_rate_flapping, OMEGA_EX_rate_flapping = 8.1, 0.05, 650.0 / 30.0


Q_TURN = 0.14  # rad/s, steady turn at 115 kt with n = 1.5, p. 475


def _run_rate_flapping(nn=1, e_over_R=E_OVER_R_EX_rate_flapping, exact_denominator=True, **ivc):
    p = om.Problem()
    p.model.add_subsystem(
        'comp', RateFlappingComp(num_nodes=nn,
                                 exact_denominator=exact_denominator),
        promotes=['*'])
    p.setup(force_alloc_complex=True)

    p.set_val('e_over_R', e_over_R)
    p.set_val('gamma', np.full(nn, GAMMA_EX_rate_flapping))
    p.set_val('Omega', np.full(nn, OMEGA_EX_rate_flapping))
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def _printed(pr, qr, mu, gamma=GAMMA_EX_rate_flapping, x=E_OVER_R_EX_rate_flapping):
    """The two expressions as printed on p. 473."""
    c16 = 16.0 / (gamma * (1 - x) ** 2)
    N1 = pr - c16 * qr
    N2 = -qr - c16 * pr
    kap = 12 * x / (gamma * (1 - x) ** 3)
    a = N1 / (1 - mu ** 2 / 2) + kap * N2 / (1 - mu ** 4 / 4)
    b = N2 / (1 + mu ** 2 / 2) - kap * N1 / (1 - mu ** 4 / 4)
    return a, b


def test_matches_printed_page_473():
    pr, qr, mu = 0.004, Q_TURN / OMEGA_EX_rate_flapping, 0.3
    p = _run_rate_flapping(exact_denominator=False, mu=mu, p=pr * OMEGA_EX_rate_flapping, q=qr * OMEGA_EX_rate_flapping)

    a, b = _printed(pr, qr, mu)
    assert_near_equal(p.get_val('a_1s_rate')[0], a, 1e-12)
    assert_near_equal(p.get_val('b_1s_rate')[0], b, 1e-12)


def test_hover_no_offset_pitch_rate_only():
    """p. 473: a_1s = -(16/gamma)(q/Omega) and b_1s = -(q/Omega)."""
    p = _run_rate_flapping(e_over_R=0.0, mu=0.0, q=Q_TURN)
    qr = Q_TURN / OMEGA_EX_rate_flapping

    assert_near_equal(p.get_val('a_1s_rate')[0], -(16 / GAMMA_EX_rate_flapping) * qr, 1e-13)
    assert_near_equal(p.get_val('b_1s_rate')[0], -qr, 1e-13)


def test_lateral_tilt_is_half_the_longitudinal_lag_at_gamma_8():
    """p. 473: for a conventional Lock number of 8, the lateral tilt is one
    half of the longitudinal lag angle."""
    p = om.Problem()
    p.model.add_subsystem('comp', RateFlappingComp(), promotes=['*'])
    p.setup()
    p.set_val('e_over_R', 0.0)
    p.set_val('gamma', 8.0)
    p.set_val('Omega', OMEGA_EX_rate_flapping)
    p.set_val('mu', 0.0)
    p.set_val('q', Q_TURN)
    p.run_model()

    ratio = p.get_val('b_1s_rate')[0] / p.get_val('a_1s_rate')[0]
    assert_near_equal(ratio, 0.5, 1e-13)


def test_nose_up_rate_tilts_the_disc_down_and_left():
    """p. 473 describes the physical signs of rate cross-coupling."""
    p = _run_rate_flapping(e_over_R=0.0, mu=0.0, q=Q_TURN)
    assert p.get_val('a_1s_rate')[0] < 0.0
    assert p.get_val('b_1s_rate')[0] < 0.0


def test_zero_rates_give_zero_flapping():
    p = _run_rate_flapping(mu=0.3, p=0.0, q=0.0)
    assert_near_equal(p.get_val('a_1s_rate')[0], 0.0, 1e-14)
    assert_near_equal(p.get_val('b_1s_rate')[0], 0.0, 1e-14)


def test_linear_in_the_rates():
    """The system is linear, so doubling p and q doubles the response."""
    one = _run_rate_flapping(mu=0.3, p=0.01, q=Q_TURN)
    two = _run_rate_flapping(mu=0.3, p=0.02, q=2 * Q_TURN)
    for name in ('a_1s_rate', 'b_1s_rate'):
        assert_near_equal(two.get_val(name)[0], 2 * one.get_val(name)[0], 1e-13)


def test_superposition_of_p_and_q():
    only_p = _run_rate_flapping(mu=0.3, p=0.01, q=0.0)
    only_q = _run_rate_flapping(mu=0.3, p=0.0, q=Q_TURN)
    both = _run_rate_flapping(mu=0.3, p=0.01, q=Q_TURN)
    for name in ('a_1s_rate', 'b_1s_rate'):
        assert_near_equal(both.get_val(name)[0],
                          only_p.get_val(name)[0] + only_q.get_val(name)[0],
                          1e-13)


def test_exact_and_simplified_denominators_are_close_rate_flapping():
    args = dict(mu=0.3, p=0.01, q=Q_TURN)
    exact = _run_rate_flapping(**args).get_val('a_1s_rate')[0]
    simple = _run_rate_flapping(exact_denominator=False, **args).get_val('a_1s_rate')[0]
    assert abs(exact / simple - 1.0) < 0.02


def test_cross_coupling_grows_with_offset():
    """kappa = 0 at e/R = 0 leaves p and q fully decoupled in the 2x2."""
    base = _run_rate_flapping(e_over_R=0.0, mu=0.3, q=Q_TURN)
    offset = _run_rate_flapping(e_over_R=0.10, mu=0.3, q=Q_TURN)
    assert abs(offset.get_val('a_1s_rate')[0]) > abs(base.get_val('a_1s_rate')[0])


def test_vectorized_rate_flapping():
    nn = 3
    mu = np.array([0.0, 0.30, 0.45])
    p = _run_rate_flapping(nn, mu=mu, p=np.full(nn, 0.01), q=np.full(nn, Q_TURN))
    assert np.all(np.isfinite(p.get_val('a_1s_rate')))
    assert np.all(p.get_val('b_1s_rate') < 0.0)



@pytest.mark.parametrize('nn,exact', [(1, True), (4, True), (4, False)])
def test_partials_rate_flapping(nn, exact):
    p = _run_rate_flapping(nn, exact_denominator=exact,
             gamma=np.linspace(6.9, 8.1, nn),
             mu=np.linspace(0.0, 0.45, nn),
             Omega=np.linspace(19.0, 21.7, nn),
             p=np.linspace(-0.02, 0.03, nn),
             q=np.linspace(0.05, 0.20, nn))
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


# ------------------------------------------------------------------------
# flapping_superposition_comp
# ------------------------------------------------------------------------

GAMMA_EX_flapping_superposition, E_OVER_R_EX_flapping_superposition = 8.1, 0.05


def _run_flapping_superposition(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', FlappingSuperpositionComp(num_nodes=nn),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_addition():
    p = _run_flapping_superposition(a_1s=0.05, b_1s=-0.012, a_1s_rate=-0.014, b_1s_rate=-0.005)
    assert_near_equal(p.get_val('a_1s_total')[0], 0.05 - 0.014, 1e-14)
    assert_near_equal(p.get_val('b_1s_total')[0], -0.012 - 0.005, 1e-14)


def test_the_two_channels_do_not_mix():
    """a_1s must not leak into b_1s_total, and vice versa."""
    only_long = _run_flapping_superposition(a_1s=0.05, a_1s_rate=-0.014)
    assert_near_equal(only_long.get_val('b_1s_total')[0], 0.0, 1e-14)

    only_lat = _run_flapping_superposition(b_1s=-0.012, b_1s_rate=-0.005)
    assert_near_equal(only_lat.get_val('a_1s_total')[0], 0.0, 1e-14)



@pytest.mark.parametrize('mu', [0.0, 0.30, 0.45])
def test_superposition_equals_a_combined_right_hand_side(mu):
    """The claim of p. 473, checked against the underlying 2x2 system.

    Steady flapping and rate flapping share the same matrix, so solving once
    with the summed right-hand side must give the same answer as adding the
    two solutions. If it did not, this component would be an approximation.
    """
    N1s, N2s = 0.031, -0.0042      # steady right-hand side
    N1r, N2r = -0.0090, -0.0037    # rate right-hand side

    kap = f2.kappa(GAMMA_EX_flapping_superposition, E_OVER_R_EX_flapping_superposition)
    det = f2.determinant(mu, kap)

    a_s, b_s = f2.solve(N1s, N2s, mu, kap, det)
    a_r, b_r = f2.solve(N1r, N2r, mu, kap, det)
    a_c, b_c = f2.solve(N1s + N1r, N2s + N2r, mu, kap, det)

    p = _run_flapping_superposition(a_1s=a_s, b_1s=b_s, a_1s_rate=a_r, b_1s_rate=b_r)
    assert_near_equal(p.get_val('a_1s_total')[0], a_c, 1e-13)
    assert_near_equal(p.get_val('b_1s_total')[0], b_c, 1e-13)


def test_pitch_rate_can_cancel_steady_flapping():
    """p. 473: in a deliberate manoeuvre the pilot uses cyclic so that both
    the longitudinal and lateral flapping are essentially zero."""
    p = _run_flapping_superposition(a_1s=0.05, b_1s=-0.012, a_1s_rate=-0.05, b_1s_rate=0.012)
    assert_near_equal(p.get_val('a_1s_total')[0], 0.0, 1e-14)
    assert_near_equal(p.get_val('b_1s_total')[0], 0.0, 1e-14)


def test_vectorized_flapping_superposition():
    nn = 3
    a = np.array([0.02, 0.05, 0.08])
    ar = np.array([-0.014, -0.015, -0.016])
    p = _run_flapping_superposition(nn, a_1s=a, a_1s_rate=ar, b_1s=np.zeros(nn),
             b_1s_rate=np.zeros(nn))
    assert_near_equal(p.get_val('a_1s_total'), a + ar, 1e-14)



@pytest.mark.parametrize('nn', [1, 4])
def test_partials_flapping_superposition(nn):
    p = _run_flapping_superposition(nn, a_1s=np.linspace(0.02, 0.08, nn),
             b_1s=np.linspace(-0.02, 0.0, nn),
             a_1s_rate=np.linspace(-0.02, -0.01, nn),
             b_1s_rate=np.linspace(-0.006, -0.002, nn))
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-12, rtol=1e-12)


def test_promoted_defaults_are_consistent_across_the_chapter():
    """Guards against ambiguous promoted inputs when the groups are combined.

    Every component that consumes mu, gamma, Omega, a or e_over_R must
    declare the same default, or OpenMDAO refuses to resolve the promoted
    value. This has caught two real mismatches already, so the guard covers
    the whole chapter rather than one pair of groups.
    """
    from prouty.flapping import (FlappingMomentsGroup,
                                 ForwardFlightFlappingGroup,
                                 HForceFlappingDerivComp, RateFlappingGroup)

    nn = 2
    p = om.Problem()
    m = p.model
    m.add_subsystem('steady', ForwardFlightFlappingGroup(num_nodes=nn),
                    promotes=['*'])
    m.add_subsystem('rates', RateFlappingGroup(num_nodes=nn), promotes=['*'])
    m.add_subsystem('moments',
                    FlappingMomentsGroup(num_nodes=nn, blade_input='external'),
                    promotes=['*'])
    m.add_subsystem('hforce', HForceFlappingDerivComp(num_nodes=nn),
                    promotes=['*'])
    p.setup()   # raises if any shared input has conflicting defaults
    p.final_setup()


# ------------------------------------------------------------------------
# maneuver_rate_comp
# ------------------------------------------------------------------------

G = 32.174


V_115KT = convert_units(115.0, 'kn', 'ft/s')


def _run_maneuver_rate(nn=1, maneuver='level_turn', units='ft/s', **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp',
                          ManeuverRateComp(num_nodes=nn, maneuver=maneuver),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in ivc.items():
        p.set_val(k, v, units=units if k == 'V' else None)
    p.run_model()
    return p


def test_example_maneuver_anchor_maneuver_rate():
    """p. 475: 115 kt, n = 1.5 gives q = 0.14 rad/s."""
    p = _run_maneuver_rate(V=V_115KT, n=1.5)
    assert_near_equal(p.get_val('q')[0], 0.14, 2e-2)


def test_formula_as_printed():
    p = _run_maneuver_rate(V=V_115KT, n=1.5)
    expected = G / V_115KT * (1.5 ** 2 - 1) / 1.5
    assert_near_equal(p.get_val('q')[0], expected, 1e-13)


def test_bank_angle_derivation():
    """q = psi_dot sin(phi) with cos(phi) = 1/n reproduces the formula."""
    for n in (1.2, 1.5, 2.0, 3.0):
        phi = np.arccos(1.0 / n)
        psi_dot = G * np.tan(phi) / V_115KT
        p = _run_maneuver_rate(V=V_115KT, n=n)
        assert_near_equal(p.get_val('q')[0], psi_dot * np.sin(phi), 1e-13)


def test_pull_up_option():
    p = _run_maneuver_rate(maneuver='pull_up', V=V_115KT, n=1.5)
    assert_near_equal(p.get_val('q')[0], G / V_115KT * 0.5, 1e-13)


def test_level_turn_needs_more_pitch_rate_than_a_pull_up():
    turn = _run_maneuver_rate(V=V_115KT, n=1.5).get_val('q')[0]
    pull = _run_maneuver_rate(maneuver='pull_up', V=V_115KT, n=1.5).get_val('q')[0]
    assert_near_equal(turn / pull, 1.667, 1e-3)



@pytest.mark.parametrize('maneuver', ['level_turn', 'pull_up'])
def test_unit_load_factor_gives_no_pitch_rate(maneuver):
    p = _run_maneuver_rate(maneuver=maneuver, V=V_115KT, n=1.0)
    assert_near_equal(p.get_val('q')[0], 0.0, 1e-14)


def test_speed_units_are_converted():
    ref = _run_maneuver_rate(V=V_115KT, n=1.5)
    kt = _run_maneuver_rate(units='kn', V=115.0, n=1.5)
    assert_near_equal(kt.get_val('q')[0], ref.get_val('q')[0], 1e-12)


def test_pitch_rate_falls_with_speed():
    nn = 3
    V = np.array([100.0, 200.0, 300.0])
    p = _run_maneuver_rate(nn, V=V, n=np.full(nn, 1.5))
    assert np.all(np.diff(p.get_val('q')) < 0)
    assert_near_equal(p.get_val('q')[0] / p.get_val('q')[2], 3.0, 1e-12)


def test_vectorized_over_load_factor():
    nn = 4
    n = np.array([1.0, 1.5, 2.0, 3.0])
    p = _run_maneuver_rate(nn, V=np.full(nn, V_115KT), n=n)
    assert_near_equal(p.get_val('q'), G / V_115KT * (n - 1 / n), 1e-13)



@pytest.mark.parametrize('nn,maneuver', [(1, 'level_turn'), (4, 'level_turn'),
                                         (4, 'pull_up')])
def test_partials_maneuver_rate(nn, maneuver):
    p = _run_maneuver_rate(nn, maneuver, V=np.linspace(120.0, 260.0, nn),
             n=np.linspace(1.1, 2.5, nn))
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_feeds_the_rate_flapping_component():
    """p. 474 to p. 473: the manoeuvre rate drives the flapping increments."""
    from prouty.flapping.rate_flapping_comp import RateFlappingComp

    p = om.Problem()
    m = p.model
    m.add_subsystem('rate', ManeuverRateComp(), promotes=['*'])
    m.add_subsystem('flap', RateFlappingComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('V', 115.0, units='kn')
    p.set_val('n', 1.5)
    p.set_val('e_over_R', 0.05)
    p.set_val('gamma', 8.1)
    p.set_val('Omega', 650.0 / 30.0)
    p.set_val('mu', 0.3)
    p.run_model()

    assert p.get_val('a_1s_rate')[0] < 0.0
    assert p.get_val('b_1s_rate')[0] < 0.0

    data = p.check_totals(of=['a_1s_rate', 'b_1s_rate'], wrt=['V', 'n'],
                          method='cs', compact_print=True, out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------------------
# lateral_cyclic_turn_comp
# ------------------------------------------------------------------------

Q_EX_lateral_cyclic_turn, OMEGA_EX_lateral_cyclic_turn, N_EX_lateral_cyclic_turn = 0.14, 21.67, 1.5


A1_LEVEL_lateral_cyclic_turn = np.radians(-2.2)


def _run_lateral_cyclic_turn(nn=1, coning_source='load_factor', **ivc):
    p = om.Problem()
    p.model.add_subsystem(
        'comp', LateralCyclicTurnComp(num_nodes=nn,
                                      coning_source=coning_source),
        promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('Omega', np.full(nn, OMEGA_EX_lateral_cyclic_turn))
    p.set_val('A_1_level', np.full(nn, A1_LEVEL_lateral_cyclic_turn))
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_example_maneuver_anchor_lateral_cyclic_turn():
    """p. 475: Delta_A_1 = 0.37 - 1.10 = -0.73 deg."""
    p = _run_lateral_cyclic_turn(q=Q_EX_lateral_cyclic_turn, n=N_EX_lateral_cyclic_turn)
    assert_near_equal(p.get_val('delta_A_1', units='deg')[0], -0.73, 1e-2)


def test_the_two_contributions_separately():
    """p. 475 splits the answer into 0.37 deg and -1.10 deg."""
    rate_only = _run_lateral_cyclic_turn(q=Q_EX_lateral_cyclic_turn, n=1.0)
    assert_near_equal(rate_only.get_val('delta_A_1', units='deg')[0],
                      0.37, 1e-2)

    coning_only = _run_lateral_cyclic_turn(q=0.0, n=N_EX_lateral_cyclic_turn)
    assert_near_equal(coning_only.get_val('delta_A_1', units='deg')[0],
                      -1.10, 1e-2)


def test_coning_wins_so_the_pilot_holds_left_stick():
    """p. 475: the effect of coning is more than the effect of pitch rate."""
    p = _run_lateral_cyclic_turn(q=Q_EX_lateral_cyclic_turn, n=N_EX_lateral_cyclic_turn)
    assert p.get_val('delta_A_1')[0] < 0.0


def test_maneuver_cyclic_is_n_times_the_level_value_plus_the_rate():
    p = _run_lateral_cyclic_turn(q=Q_EX_lateral_cyclic_turn, n=N_EX_lateral_cyclic_turn)
    assert_near_equal(p.get_val('A_1_maneuver')[0],
                      N_EX_lateral_cyclic_turn * A1_LEVEL_lateral_cyclic_turn + Q_EX_lateral_cyclic_turn / OMEGA_EX_lateral_cyclic_turn, 1e-13)


def test_level_flight_needs_no_change():
    p = _run_lateral_cyclic_turn(q=0.0, n=1.0)
    assert_near_equal(p.get_val('delta_A_1')[0], 0.0, 1e-14)
    assert_near_equal(p.get_val('A_1_maneuver')[0], A1_LEVEL_lateral_cyclic_turn, 1e-14)


def test_direct_route_matches_the_chapter_3_relation():
    """p. 474: A_1 - b_1s = -[(4/3) mu a_0 + v_1] / (1 + mu^2/2), b_1s = 0."""
    mu, a_0, v1 = 0.3, np.radians(3.2), 0.0117
    p = _run_lateral_cyclic_turn(coning_source='direct', q=Q_EX_lateral_cyclic_turn, mu=mu, a_0=a_0,
             v1_over_OmegaR=v1)

    expected = Q_EX_lateral_cyclic_turn / OMEGA_EX_lateral_cyclic_turn - ((4 / 3) * mu * a_0 + v1) / (1 + mu ** 2 / 2)
    assert_near_equal(p.get_val('A_1_maneuver')[0], expected, 1e-13)


def test_load_factor_shortcut_understates_the_coning_change():
    """The blade weight term of a_0 does not scale with thrust.

    The error left in the numerator is exactly (4/3) mu a_0,weight (n - 1),
    which is 1.3 % of A_1 for the example helicopter at n = 1.5.
    """
    mu, v1_level = 0.3, 0.0117
    a0_aero, a0_weight = np.radians(3.39), np.radians(0.19)

    def A1(a_0, v1):
        return -((4 / 3) * mu * a_0 + v1) / (1 + mu ** 2 / 2)

    level = A1(a0_aero - a0_weight, v1_level)
    true_man = A1(N_EX_lateral_cyclic_turn * a0_aero - a0_weight, N_EX_lateral_cyclic_turn * v1_level)
    shortcut = N_EX_lateral_cyclic_turn * level

    gap = (4 / 3) * mu * a0_weight * (N_EX_lateral_cyclic_turn - 1) / (1 + mu ** 2 / 2)
    assert_near_equal(shortcut - true_man, gap, 1e-12)

    assert abs(true_man) > abs(shortcut)
    assert_near_equal(abs(true_man / shortcut - 1.0), 0.013, 5e-2)


def test_pitch_rate_term_cancels_the_rate_flapping():
    """Delta_A_1 = q/Omega is exactly what cancels b_1s from pitch rate.

    In hover with no offset, RateFlappingComp gives b_1s = -(q/Omega), and
    the cosine equation has db_1s/dA_1 = 1.
    """
    from prouty.flapping.rate_flapping_comp import RateFlappingComp

    p = om.Problem()
    p.model.add_subsystem('flap', RateFlappingComp(), promotes=['*'])
    p.setup()
    p.set_val('e_over_R', 0.0)
    p.set_val('gamma', 8.1)
    p.set_val('Omega', OMEGA_EX_lateral_cyclic_turn)
    p.set_val('mu', 0.0)
    p.set_val('q', Q_EX_lateral_cyclic_turn)
    p.run_model()

    correction = _run_lateral_cyclic_turn(q=Q_EX_lateral_cyclic_turn, n=1.0).get_val('delta_A_1')[0]
    assert_near_equal(correction, -p.get_val('b_1s_rate')[0], 1e-13)


def test_vectorized_lateral_cyclic_turn():
    nn = 4
    n = np.array([1.0, 1.25, 1.5, 2.0])
    p = _run_lateral_cyclic_turn(nn, q=np.full(nn, Q_EX_lateral_cyclic_turn), n=n)
    assert np.all(np.diff(p.get_val('delta_A_1')) < 0)



@pytest.mark.parametrize('nn,source', [(1, 'load_factor'), (4, 'load_factor'),
                                       (4, 'direct')])
def test_partials_lateral_cyclic_turn(nn, source):
    extra = ({'n': np.linspace(1.1, 2.0, nn)} if source == 'load_factor'
             else {'mu': np.linspace(0.15, 0.45, nn),
                   'a_0': np.radians(np.linspace(2.8, 3.6, nn)),
                   'v1_over_OmegaR': np.linspace(0.008, 0.015, nn)})
    p = _run_lateral_cyclic_turn(nn, source, q=np.linspace(0.05, 0.25, nn),
             Omega=np.linspace(19.0, 21.7, nn),
             A_1_level=np.radians(np.linspace(-3.0, -1.5, nn)), **extra)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_chained_from_the_maneuver_rate_lateral_cyclic_turn():
    from prouty.flapping.maneuver_rate_comp import ManeuverRateComp

    p = om.Problem()
    m = p.model
    m.add_subsystem('rate', ManeuverRateComp(), promotes=['*'])
    m.add_subsystem('cyclic', LateralCyclicTurnComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('V', 115.0, units='kn')
    p.set_val('n', N_EX_lateral_cyclic_turn)
    p.set_val('Omega', OMEGA_EX_lateral_cyclic_turn)
    p.set_val('A_1_level', A1_LEVEL_lateral_cyclic_turn)
    p.run_model()

    assert_near_equal(p.get_val('delta_A_1', units='deg')[0], -0.73, 2e-2)

    data = p.check_totals(of=['delta_A_1', 'A_1_maneuver'],
                          wrt=['V', 'n', 'Omega', 'A_1_level'],
                          method='cs', compact_print=True, out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------------------
# longitudinal_cyclic_turn_comp
# ------------------------------------------------------------------------

Q_EX_longitudinal_cyclic_turn, OMEGA_EX_longitudinal_cyclic_turn, GAMMA_EX_longitudinal_cyclic_turn, MU_EX_longitudinal_cyclic_turn = 0.14, 21.67, 8.1, 0.3


def _a1s_rate_hover(q=Q_EX_longitudinal_cyclic_turn, gamma=GAMMA_EX_longitudinal_cyclic_turn, Omega=OMEGA_EX_longitudinal_cyclic_turn):
    """p. 473, hover with no offset: a_1s = -(16/gamma)(q/Omega)."""
    return -(16.0 / gamma) * q / Omega


def _run_longitudinal_cyclic_turn(nn=1, exact_denominator=True, e_over_R=0.0, **ivc):
    p = om.Problem()
    p.model.add_subsystem(
        'comp',
        LongitudinalCyclicTurnComp(num_nodes=nn,
                                   exact_denominator=exact_denominator),
        promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('e_over_R', e_over_R)
    p.set_val('gamma', np.full(nn, GAMMA_EX_longitudinal_cyclic_turn))
    p.set_val('mu', np.full(nn, MU_EX_longitudinal_cyclic_turn))
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_example_maneuver_anchor_longitudinal_cyclic_turn():
    """p. 475: Delta_B_1 = -0.62 deg."""
    p = _run_longitudinal_cyclic_turn(a_1s_rate=_a1s_rate_hover())
    assert_near_equal(p.get_val('delta_B_1', units='deg')[0], -0.62, 1e-2)


def test_sensitivity_matches_the_printed_trim_derivative():
    """p. 475: da_1s/dB_1 = -(1 + 3 mu^2/2) / (1 - mu^2/2)."""
    for mu in (0.0, 0.15, 0.30, 0.45):
        p = _run_longitudinal_cyclic_turn(mu=mu)
        expected = -(1 + 1.5 * mu ** 2) / (1 - mu ** 2 / 2)
        assert_near_equal(p.get_val('da1s_dB1')[0], expected, 1e-13)


def test_printed_formula_reproduced():
    """Delta_B_1 = -[(1 - mu^2/2)/(1 + 3 mu^2/2)] (16/gamma)(q/Omega).

    Reproduced by feeding the *hover* value of a_1s,rate, which is what the
    printed numerator is. See test_printed_numerator_is_the_hover_value.
    """
    p = _run_longitudinal_cyclic_turn(a_1s_rate=_a1s_rate_hover())
    expected = -((1 - MU_EX_longitudinal_cyclic_turn ** 2 / 2) / (1 + 1.5 * MU_EX_longitudinal_cyclic_turn ** 2)) \
        * (16 / GAMMA_EX_longitudinal_cyclic_turn) * Q_EX_longitudinal_cyclic_turn / OMEGA_EX_longitudinal_cyclic_turn
    assert_near_equal(p.get_val('delta_B_1')[0], expected, 1e-13)


def test_printed_numerator_is_the_hover_value():
    """Entry C7-10: p. 475 mixes a hover numerator with a forward flight
    denominator.

    p. 473 gives a_1s,rate = -(16/gamma)(q/Omega) / (1 - mu^2/2) at zero
    offset. p. 475 uses the numerator alone, leaving out 1/(1 - mu^2/2).
    """
    for mu in (0.15, 0.30, 0.45):
        hover = _a1s_rate_hover()
        consistent = hover / (1 - mu ** 2 / 2)

        printed = _run_longitudinal_cyclic_turn(mu=mu, a_1s_rate=hover).get_val('delta_B_1')[0]
        proper = _run_longitudinal_cyclic_turn(mu=mu, a_1s_rate=consistent).get_val('delta_B_1')[0]

        assert_near_equal(proper / printed, 1 / (1 - mu ** 2 / 2), 1e-13)


def test_cyclic_cancels_the_rate_flapping():
    """The whole point: (da_1s/dB_1) Delta_B_1 must undo a_1s,rate."""
    a_r = _a1s_rate_hover()
    p = _run_longitudinal_cyclic_turn(a_1s_rate=a_r)
    residual = p.get_val('da1s_dB1')[0] * p.get_val('delta_B_1')[0] + a_r
    assert_near_equal(residual, 0.0, 1e-15)


def test_nose_up_rate_needs_forward_stick():
    p = _run_longitudinal_cyclic_turn(a_1s_rate=_a1s_rate_hover())
    assert p.get_val('delta_B_1')[0] < 0.0


def test_no_rate_needs_no_cyclic():
    p = _run_longitudinal_cyclic_turn(a_1s_rate=0.0)
    assert_near_equal(p.get_val('delta_B_1')[0], 0.0, 1e-15)


def test_offset_produces_lateral_flapping_through_B1():
    """db_1s/dB_1 = kappa (1 + 3 mu^2/2) / Delta, zero without offset."""
    none = _run_longitudinal_cyclic_turn(e_over_R=0.0)
    assert_near_equal(none.get_val('db1s_dB1')[0], 0.0, 1e-15)

    with_offset = _run_longitudinal_cyclic_turn(e_over_R=0.05)
    kap = 12 * 0.05 / (GAMMA_EX_longitudinal_cyclic_turn * (1 - 0.05) ** 3)
    det = 1 - MU_EX_longitudinal_cyclic_turn ** 4 / 4 + kap ** 2
    assert_near_equal(with_offset.get_val('db1s_dB1')[0],
                      kap * (1 + 1.5 * MU_EX_longitudinal_cyclic_turn ** 2) / det, 1e-13)


def test_exact_and_simplified_denominators_are_close_longitudinal_cyclic_turn():
    args = dict(a_1s_rate=_a1s_rate_hover(), e_over_R=0.05)
    exact = _run_longitudinal_cyclic_turn(**args).get_val('delta_B_1')[0]
    simple = _run_longitudinal_cyclic_turn(exact_denominator=False, **args).get_val('delta_B_1')[0]
    assert abs(exact / simple - 1.0) < 0.02


def test_coupling_is_almost_one_to_one_longitudinal_cyclic_turn():
    """p. 475: Delta_B_1 = -0.62 deg against Delta_A_1 = -0.73 deg."""
    from prouty.flapping.lateral_cyclic_turn_comp import LateralCyclicTurnComp

    lat = om.Problem()
    lat.model.add_subsystem('c', LateralCyclicTurnComp(), promotes=['*'])
    lat.setup()
    lat.set_val('q', Q_EX_longitudinal_cyclic_turn)
    lat.set_val('Omega', OMEGA_EX_longitudinal_cyclic_turn)
    lat.set_val('n', 1.5)
    lat.set_val('A_1_level', np.radians(-2.2))
    lat.run_model()

    lon = _run_longitudinal_cyclic_turn(a_1s_rate=_a1s_rate_hover())

    ratio = (lon.get_val('delta_B_1')[0] / lat.get_val('delta_A_1')[0])
    assert 0.7 < ratio < 1.2


def test_vectorized_longitudinal_cyclic_turn():
    nn = 4
    mu = np.array([0.0, 0.15, 0.30, 0.45])
    p = _run_longitudinal_cyclic_turn(nn, mu=mu, a_1s_rate=np.full(nn, _a1s_rate_hover()))
    assert np.all(np.diff(np.abs(p.get_val('delta_B_1'))) < 0)



@pytest.mark.parametrize('nn,exact', [(1, True), (4, True), (4, False)])
def test_partials_longitudinal_cyclic_turn(nn, exact):
    p = _run_longitudinal_cyclic_turn(nn, exact_denominator=exact, e_over_R=0.05,
             a_1s_rate=np.linspace(-0.020, -0.005, nn),
             mu=np.linspace(0.10, 0.45, nn),
             gamma=np.linspace(6.9, 8.1, nn))
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_chained_from_the_maneuver_rate_longitudinal_cyclic_turn():
    from prouty.flapping.maneuver_rate_comp import ManeuverRateComp
    from prouty.flapping.rate_flapping_comp import RateFlappingComp

    p = om.Problem()
    m = p.model
    m.add_subsystem('rate', ManeuverRateComp(), promotes=['*'])
    m.add_subsystem('flap', RateFlappingComp(), promotes=['*'])
    m.add_subsystem('cyclic', LongitudinalCyclicTurnComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('V', 115.0, units='kn')
    p.set_val('n', 1.5)
    p.set_val('Omega', OMEGA_EX_longitudinal_cyclic_turn)
    p.set_val('gamma', GAMMA_EX_longitudinal_cyclic_turn)
    p.set_val('mu', MU_EX_longitudinal_cyclic_turn)
    p.set_val('e_over_R', 0.0)
    p.run_model()

    # -0.636 deg, not the -0.62 of p. 475: the chain feeds the a_1s,rate that
    # actually applies at mu = 0.3, where the book uses its hover value.
    assert_near_equal(p.get_val('delta_B_1', units='deg')[0],
                      -0.62 / (1 - MU_EX_longitudinal_cyclic_turn ** 2 / 2), 3e-2)

    data = p.check_totals(of=['delta_B_1'], wrt=['V', 'n', 'Omega', 'gamma',
                                                 'mu'],
                          method='cs', compact_print=True, out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------------------
# rate_flapping_group
# ------------------------------------------------------------------------

OMEGA_EX_rate_flapping_grp, GAMMA_EX_rate_flapping_grp, MU_EX_rate_flapping_grp, N_EX_rate_flapping_grp = 21.67, 8.1, 0.3, 1.5


A1_LEVEL_rate_flapping_grp = np.radians(-2.2)


def _build(nn=1, e_over_R=0.0, **opts):
    over = {k: opts.pop(k) for k in list(opts) if k in
            ('V', 'n', 'p', 'q', 'mu', 'a_1s', 'b_1s', 'a_0',
             'v1_over_OmegaR')}

    p = om.Problem()
    p.model.add_subsystem('g3', RateFlappingGroup(num_nodes=nn, **opts),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)

    p.set_val('e_over_R', e_over_R)
    p.set_val('gamma', np.full(nn, GAMMA_EX_rate_flapping_grp))
    p.set_val('Omega', np.full(nn, OMEGA_EX_rate_flapping_grp))
    p.set_val('mu', np.full(nn, MU_EX_rate_flapping_grp))
    p.set_val('A_1_level', np.full(nn, A1_LEVEL_rate_flapping_grp))
    if opts.get('rate_source', 'maneuver') == 'maneuver':
        p.set_val('V', np.full(nn, 115.0), units='kn')
    if opts.get('coning_source', 'load_factor') == 'load_factor':
        p.set_val('n', np.full(nn, N_EX_rate_flapping_grp))
    for k, v in over.items():
        p.set_val(k, v)

    p.run_model()
    return p


def test_example_maneuver_end_to_end():
    """The p. 475 worked example, from airspeed and load factor."""
    p = _build()

    assert_near_equal(p.get_val('q')[0], 0.14, 2e-2)
    assert_near_equal(p.get_val('delta_A_1', units='deg')[0], -0.73, 2e-2)

    # delta_B_1 is -0.644 rather than -0.62, see entry C7-10
    assert_near_equal(p.get_val('delta_B_1', units='deg')[0],
                      -0.62 / (1 - MU_EX_rate_flapping_grp ** 2 / 2), 3e-2)


def test_coupling_is_almost_one_to_one_rate_flapping_grp():
    """p. 475: the pilot sees comparable lateral and longitudinal stick."""
    p = _build()
    ratio = p.get_val('delta_B_1')[0] / p.get_val('delta_A_1')[0]
    assert 0.7 < ratio < 1.2


def test_external_rate_source_reproduces_the_maneuver():
    ref = _build()
    ext = _build(rate_source='external', q=ref.get_val('q'))

    for name in ('a_1s_rate', 'b_1s_rate', 'delta_A_1', 'delta_B_1'):
        assert_near_equal(ext.get_val(name)[0], ref.get_val(name)[0], 1e-13)


def test_roll_rate_only_flows_through():
    """A roll rate produces longitudinal flapping through the p/Omega term of
    N_1 (p. 473), so it calls for longitudinal cyclic even with q = 0.

    Taking a_1s_rate as the input rather than rebuilding (16/gamma)(q/Omega)
    is what makes that work; the printed p. 475 formula sees only q.
    Delta_A_1 does stay zero, being driven by q and the load factor alone.
    """
    roll = 0.05
    p = _build(rate_source='external', q=0.0, p=roll, n=1.0)

    assert p.get_val('a_1s_rate')[0] > 0.0
    assert p.get_val('b_1s_rate')[0] < 0.0
    assert_near_equal(p.get_val('delta_A_1')[0], 0.0, 1e-14)

    # a_1s_rate = (p/Omega) / (1 - mu^2/2) at zero offset
    assert_near_equal(p.get_val('a_1s_rate')[0],
                      roll / OMEGA_EX_rate_flapping_grp / (1 - MU_EX_rate_flapping_grp ** 2 / 2), 1e-13)
    assert p.get_val('delta_B_1')[0] > 0.0


def test_superposition_adds_the_steady_flapping():
    steady_a, steady_b = np.radians(2.87), np.radians(-0.78)
    p = _build(a_1s=steady_a, b_1s=steady_b)

    assert_near_equal(p.get_val('a_1s_total')[0],
                      steady_a + p.get_val('a_1s_rate')[0], 1e-14)
    assert_near_equal(p.get_val('b_1s_total')[0],
                      steady_b + p.get_val('b_1s_rate')[0], 1e-14)


def test_unconnected_steady_flapping_defaults_to_zero():
    p = _build()
    assert_near_equal(p.get_val('a_1s_total')[0],
                      p.get_val('a_1s_rate')[0], 1e-15)


def test_level_flight_is_quiescent():
    p = _build(n=1.0)
    for name in ('q', 'a_1s_rate', 'b_1s_rate', 'delta_A_1', 'delta_B_1'):
        assert_near_equal(p.get_val(name)[0], 0.0, 1e-14)


def test_pull_up_needs_less_cyclic_than_a_level_turn():
    turn = _build().get_val('delta_B_1')[0]
    pull = _build(maneuver='pull_up').get_val('delta_B_1')[0]
    assert abs(pull) < abs(turn)
    # q_turn / q_pull = [(n^2-1)/n] / (n-1) = (n+1)/n
    assert_near_equal(turn / pull, (N_EX_rate_flapping_grp + 1.0) / N_EX_rate_flapping_grp, 1e-12)


def test_direct_coning_source():
    p = _build(coning_source='direct', a_0=np.radians(4.8),
               v1_over_OmegaR=0.0176)

    expected = p.get_val('q')[0] / OMEGA_EX_rate_flapping_grp - (
        (4 / 3) * MU_EX_rate_flapping_grp * np.radians(4.8) + 0.0176) / (1 + MU_EX_rate_flapping_grp ** 2 / 2)
    assert_near_equal(p.get_val('A_1_maneuver')[0], expected, 1e-13)


def test_offset_creates_control_cross_coupling():
    """Suppressing a_1s with B_1 also rolls the disc once e/R is nonzero."""
    assert_near_equal(_build(e_over_R=0.0).get_val('db1s_dB1')[0], 0.0, 1e-15)
    assert _build(e_over_R=0.05).get_val('db1s_dB1')[0] > 0.0


def test_group_is_feed_forward():
    p = _build(nn=2, V=np.full(2, 115.0), n=np.full(2, N_EX_rate_flapping_grp))
    before = p.get_val('delta_B_1').copy()
    p.run_model()
    assert_near_equal(p.get_val('delta_B_1'), before, 1e-14)



@pytest.mark.parametrize('rate_source', ['maneuver', 'external'])
def test_totals(rate_source):
    nn = 3
    extra = ({'V': np.full(nn, 115.0)} if rate_source == 'maneuver'
             else {'q': np.linspace(0.05, 0.20, nn),
                   'p': np.linspace(-0.02, 0.03, nn)})
    p = _build(nn, e_over_R=0.05, rate_source=rate_source,
               mu=np.array([0.15, 0.30, 0.45]), n=np.full(nn, N_EX_rate_flapping_grp), **extra)

    wrt = ['e_over_R', 'gamma', 'Omega', 'mu', 'n', 'A_1_level', 'a_1s', 'b_1s']
    wrt += ['V'] if rate_source == 'maneuver' else ['p', 'q']

    data = p.check_totals(
        of=['a_1s_rate', 'b_1s_rate', 'a_1s_total', 'b_1s_total',
            'delta_A_1', 'delta_B_1'],
        wrt=wrt, method='cs', compact_print=True, out_stream=None)

    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-8, atol=1e-11)


def test_chains_with_the_forward_flight_group():
    """G2 into G3: promoted names and defaults must line up."""
    from prouty.flapping import ForwardFlightFlappingGroup

    nn = 2
    p = om.Problem()
    m = p.model
    m.add_subsystem('steady', ForwardFlightFlappingGroup(num_nodes=nn),
                    promotes=['*'])
    m.add_subsystem('rates', RateFlappingGroup(num_nodes=nn), promotes=['*'])
    p.setup()

    p.set_val('e_over_R', 0.05)
    p.set_val('R', 30.0)
    p.set_val('I_b_ref', 2870.0)
    p.set_val('sigma', 0.08488)
    p.set_val('a', np.full(nn, 6.0))
    p.set_val('gamma', np.full(nn, GAMMA_EX_rate_flapping_grp))
    p.set_val('Omega', np.full(nn, OMEGA_EX_rate_flapping_grp))
    p.set_val('mu', np.array([0.2, 0.35]))
    p.set_val('theta_0', np.full(nn, np.radians(14.0)))
    p.set_val('theta_1', np.full(nn, np.radians(-10.0)))
    p.set_val('alpha_s', np.full(nn, np.radians(-5.0)))
    p.set_val('A_1', np.full(nn, A1_LEVEL_rate_flapping_grp))
    p.set_val('B_1', np.full(nn, np.radians(1.5)))
    p.set_val('V', np.full(nn, 115.0), units='kn')
    p.set_val('n', np.full(nn, N_EX_rate_flapping_grp))
    p.set_val('A_1_level', np.full(nn, A1_LEVEL_rate_flapping_grp))
    p.run_model()

    assert_near_equal(p.get_val('a_1s_total'),
                      p.get_val('a_1s') + p.get_val('a_1s_rate'), 1e-14)
    assert np.all(p.get_val('a_1s_rate') < 0.0)
