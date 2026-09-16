"""Chapter 7 -- H-force due to flapping, p. 478-479."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.assert_utils import assert_near_equal

from prouty.flapping import HForceFlappingGroup
from prouty.flapping.flapping_moment_buildup_comp import \
    FlappingMomentBuildupComp
from prouty.flapping.h_force_flapping_deriv_comp import HForceFlappingDerivComp


# ------------------------------------------------------------------------
# h_force_flapping_deriv_comp
# ------------------------------------------------------------------------

GW_h_force_flapping_deriv, R_EX_h_force_flapping_deriv, OMEGA_R_h_force_flapping_deriv, SIGMA_h_force_flapping_deriv, A_EX_h_force_flapping_deriv, H_M_h_force_flapping_deriv = 20000.0, 30.0, 650.0, 0.085, 6.0, 7.5


RHO_SL_h_force_flapping_deriv = 0.002378


DISK_Q = RHO_SL_h_force_flapping_deriv * np.pi * R_EX_h_force_flapping_deriv ** 2 * OMEGA_R_h_force_flapping_deriv ** 2


CT_h_force_flapping_deriv = GW_h_force_flapping_deriv / DISK_Q


CT_SIGMA_h_force_flapping_deriv = CT_h_force_flapping_deriv / SIGMA_h_force_flapping_deriv


def _run_h_force_flapping_deriv(nn=1, form='lambda', **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp',
                          HForceFlappingDerivComp(num_nodes=nn, form=form),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('CT_sigma', np.full(nn, CT_SIGMA_h_force_flapping_deriv))
    p.set_val('a', np.full(nn, A_EX_h_force_flapping_deriv))
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_lambda_form_as_printed():
    """d(C_H/sigma)/da_1s = C_T/sigma + (a/8) lambda', p. 479."""
    mu, als, a1s, v1 = 0.3, np.radians(-5.0), np.radians(2.87), 0.0117
    p = _run_h_force_flapping_deriv(mu=mu, alpha_s=als, a_1s=a1s, v1_over_OmegaR=v1)

    lam = mu * (als + a1s) - v1
    assert_near_equal(p.get_val('lambda_prime')[0], lam, 1e-13)
    assert_near_equal(p.get_val('dCHsigma_da1s')[0],
                      CT_SIGMA_h_force_flapping_deriv + A_EX_h_force_flapping_deriv / 8 * lam, 1e-13)


def test_lambda_prime_matches_the_chapter_3_definition():
    """p. 166: lambda' = lambda + mu(B_1 + a_1s) with
    lambda = mu(alpha_s - B_1) - v_1/(Omega R).
    """
    mu, als, a1s, v1, B1 = 0.3, np.radians(-5.0), np.radians(2.87), 0.0117, \
        np.radians(1.5)
    p = _run_h_force_flapping_deriv(mu=mu, alpha_s=als, a_1s=a1s, v1_over_OmegaR=v1)

    lam_cp = mu * (als - B1) - v1
    assert_near_equal(p.get_val('lambda_prime')[0], lam_cp + mu * (B1 + a1s),
                      1e-13)


def test_y_force_derivative_equals_the_h_force_one():
    """p. 479: the Y-force behaves exactly as the H-force."""
    p = _run_h_force_flapping_deriv(mu=0.3, alpha_s=np.radians(-5.0), a_1s=np.radians(2.87),
             v1_over_OmegaR=0.0117)
    assert_near_equal(p.get_val('dCYsigma_db1s')[0],
                      p.get_val('dCHsigma_da1s')[0], 1e-14)


def test_theta75_form_as_printed():
    """(3/2)(C_T/sigma)[1 - (a/18) theta_75 / (C_T/sigma)], p. 479."""
    th75 = 0.1719
    p = _run_h_force_flapping_deriv(form='theta75', theta_75=th75)

    printed = 1.5 * CT_SIGMA_h_force_flapping_deriv * (1 - A_EX_h_force_flapping_deriv / 18 * th75 / CT_SIGMA_h_force_flapping_deriv)
    assert_near_equal(p.get_val('dCHsigma_da1s')[0], printed, 1e-13)


def test_the_two_forms_agree_in_hover():
    """They are the same expression given C_T/sigma = (a/4)((2/3) th75 + l')."""
    v1 = np.sqrt(CT_h_force_flapping_deriv / 2.0)
    lam = -v1                                   # hover: mu = 0
    th75 = 1.5 * (4 * CT_SIGMA_h_force_flapping_deriv / A_EX_h_force_flapping_deriv - lam)    # invert the thrust relation

    by_lambda = _run_h_force_flapping_deriv(mu=0.0, alpha_s=0.0, a_1s=0.0, v1_over_OmegaR=v1)
    by_theta = _run_h_force_flapping_deriv(form='theta75', theta_75=th75)

    assert_near_equal(by_theta.get_val('dCHsigma_da1s')[0],
                      by_lambda.get_val('dCHsigma_da1s')[0], 1e-13)


def test_hover_anchor_against_the_page_479_table():
    """p. 479 gives 73,500 ft-lb/rad from the rotor force in hover.

    With h_M = 7.5 ft that is a force derivative of 9,800 lb. The closed form
    gives 9,255, 5.6 % low: the table's theta_75 comes from the full Chapter 1
    hover analysis, not from the idealised momentum value used here.
    """
    v1 = np.sqrt(CT_h_force_flapping_deriv / 2.0)
    p = _run_h_force_flapping_deriv(mu=0.0, alpha_s=0.0, a_1s=0.0, v1_over_OmegaR=v1)

    force = p.get_val('dCHsigma_da1s')[0] * SIGMA_h_force_flapping_deriv * DISK_Q
    assert_near_equal(force, 9255.0, 1e-3)
    assert_near_equal(force * H_M_h_force_flapping_deriv, 73500.0, 6e-2)


def test_inflow_more_than_halves_the_perpendicular_assumption():
    """The simple T a_1s of p. 476 against the corrected derivative."""
    v1 = np.sqrt(CT_h_force_flapping_deriv / 2.0)
    p = _run_h_force_flapping_deriv(mu=0.0, alpha_s=0.0, a_1s=0.0, v1_over_OmegaR=v1)

    force = p.get_val('dCHsigma_da1s')[0] * SIGMA_h_force_flapping_deriv * DISK_Q
    assert_near_equal(GW_h_force_flapping_deriv / force, 2.16, 1e-2)


def test_derivative_falls_as_lambda_prime_becomes_more_negative():
    """p. 479: the effect of the tilt of the thrust vector is reduced and in
    some extreme conditions might even change sign."""
    values = [_run_h_force_flapping_deriv(mu=0.3, alpha_s=np.radians(A), a_1s=0.0,
                   v1_over_OmegaR=0.0117).get_val('dCHsigma_da1s')[0]
              for A in (0.0, -5.0, -10.0, -20.0)]
    assert all(b < a for a, b in zip(values, values[1:]))

    steep = _run_h_force_flapping_deriv(mu=0.5, alpha_s=np.radians(-25.0), a_1s=0.0,
                 v1_over_OmegaR=0.02)
    assert steep.get_val('dCHsigma_da1s')[0] < 0.0


def test_the_dropped_term_is_a_third_of_the_answer():
    """Entry C7-11. The step from the full derivative to C_T/sigma + (a/8)l'
    drops exactly -(a/4) mu a_1s.
    """
    mu, a1s = 0.3, np.radians(2.87)
    p = _run_h_force_flapping_deriv(mu=mu, alpha_s=np.radians(-5.0), a_1s=a1s, v1_over_OmegaR=0.0117)

    dropped = A_EX_h_force_flapping_deriv * mu * a1s / 4.0
    kept = p.get_val('dCHsigma_da1s')[0]

    assert_near_equal(dropped, 0.0225, 1e-2)
    assert 0.25 < dropped / abs(kept) < 0.45


def test_vectorized_h_force_flapping_deriv():
    nn = 3
    mu = np.array([0.0, 0.3, 0.45])
    p = _run_h_force_flapping_deriv(nn, mu=mu, alpha_s=np.full(nn, np.radians(-5.0)),
             a_1s=np.zeros(nn), v1_over_OmegaR=np.full(nn, 0.0117))
    assert np.all(np.diff(p.get_val('lambda_prime')) < 0)
    assert np.all(np.diff(p.get_val('dCHsigma_da1s')) < 0)



@pytest.mark.parametrize('nn,form', [(1, 'lambda'), (4, 'lambda'),
                                     (4, 'theta75')])
def test_partials_h_force_flapping_deriv(nn, form):
    extra = ({'theta_75': np.linspace(0.10, 0.20, nn)} if form == 'theta75'
             else {'mu': np.linspace(0.1, 0.45, nn),
                   'alpha_s': np.radians(np.linspace(-8.0, -2.0, nn)),
                   'a_1s': np.radians(np.linspace(0.0, 4.0, nn)),
                   'v1_over_OmegaR': np.linspace(0.008, 0.06, nn)})
    p = _run_h_force_flapping_deriv(nn, form, CT_sigma=np.linspace(0.06, 0.09, nn),
             a=np.linspace(5.7, 6.1, nn), **extra)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-10, rtol=1e-10)


def test_chained_from_the_forward_flight_group():
    from prouty.flapping import ForwardFlightFlappingGroup

    nn = 2
    p = om.Problem()
    m = p.model
    m.add_subsystem('steady', ForwardFlightFlappingGroup(num_nodes=nn),
                    promotes=['*'])
    m.add_subsystem('hforce', HForceFlappingDerivComp(num_nodes=nn),
                    promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('e_over_R', 0.05)
    p.set_val('R', R_EX_h_force_flapping_deriv)
    p.set_val('I_b_ref', 2870.0)
    p.set_val('sigma', SIGMA_h_force_flapping_deriv)
    p.set_val('a', np.full(nn, A_EX_h_force_flapping_deriv))
    p.set_val('gamma', np.full(nn, 8.1))
    p.set_val('Omega', np.full(nn, OMEGA_R_h_force_flapping_deriv / R_EX_h_force_flapping_deriv))
    p.set_val('mu', np.array([0.2, 0.4]))
    p.set_val('theta_0', np.full(nn, np.radians(14.0)))
    p.set_val('theta_1', np.full(nn, np.radians(-10.0)))
    p.set_val('alpha_s', np.full(nn, np.radians(-5.0)))
    p.set_val('A_1', np.full(nn, np.radians(-2.2)))
    p.set_val('B_1', np.full(nn, np.radians(1.5)))
    p.run_model()

    assert np.all(p.get_val('lambda_prime') < 0.0)
    assert np.all(p.get_val('dCHsigma_da1s') < p.get_val('CT_sigma'))

    data = p.check_totals(of=['dCHsigma_da1s', 'lambda_prime'],
                          wrt=['mu', 'theta_0', 'alpha_s', 'sigma', 'a'],
                          method='cs', compact_print=True, out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-9, atol=1e-12)


# ------------------------------------------------------------------------
# flapping_moment_buildup_comp
# ------------------------------------------------------------------------

GW_flapping_moment_buildup, R_EX_flapping_moment_buildup, OMEGA_R_flapping_moment_buildup, SIGMA_flapping_moment_buildup, A_EX_flapping_moment_buildup, H_M_flapping_moment_buildup = 20000.0, 30.0, 650.0, 0.085, 6.0, 7.5


RHO_SL_flapping_moment_buildup, OMEGA_EX_flapping_moment_buildup = 0.002378, 650.0 / 30.0


SCALE_flapping_moment_buildup = SIGMA_flapping_moment_buildup * RHO_SL_flapping_moment_buildup * np.pi * R_EX_flapping_moment_buildup ** 4 * OMEGA_EX_flapping_moment_buildup ** 2


CT_flapping_moment_buildup = GW_flapping_moment_buildup / (RHO_SL_flapping_moment_buildup * np.pi * R_EX_flapping_moment_buildup ** 2 * OMEGA_R_flapping_moment_buildup ** 2)


CT_SIGMA_flapping_moment_buildup = CT_flapping_moment_buildup / SIGMA_flapping_moment_buildup


K_HUB_flapping_moment_buildup = 200940.0   # p. 477


TABLE = {'hover': (73500.0, 274440.0),
         '115kt': (123000.0, 323940.0),
         '160kt': (108000.0, 308940.0)}


def _run_flapping_moment_buildup(nn=1, **ivc):
    p = om.Problem()
    p.model.add_subsystem('comp', FlappingMomentBuildupComp(num_nodes=nn),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('R', R_EX_flapping_moment_buildup)
    p.set_val('sigma', SIGMA_flapping_moment_buildup)
    p.set_val('h_M', H_M_flapping_moment_buildup)
    p.set_val('rho', np.full(nn, RHO_SL_flapping_moment_buildup))
    p.set_val('Omega', np.full(nn, OMEGA_EX_flapping_moment_buildup))
    p.set_val('dMM_da1s', np.full(nn, K_HUB_flapping_moment_buildup))
    for k, v in ivc.items():
        p.set_val(k, v)
    p.run_model()
    return p


def _dCH_for(force_contribution):
    """The coefficient derivative a table entry implies."""
    return force_contribution / H_M_flapping_moment_buildup / SCALE_flapping_moment_buildup


def test_force_contribution_definition():
    dCH = 0.0384
    p = _run_flapping_moment_buildup(dCHsigma_da1s=dCH)
    assert_near_equal(p.get_val('dMCG_da1s_force')[0], dCH * SCALE_flapping_moment_buildup * H_M_flapping_moment_buildup, 1e-13)


def test_total_adds_the_hub_couple():
    dCH = 0.0384
    p = _run_flapping_moment_buildup(dCHsigma_da1s=dCH)
    assert_near_equal(p.get_val('dMCG_da1s')[0],
                      p.get_val('dMCG_da1s_force')[0] + K_HUB_flapping_moment_buildup, 1e-13)



@pytest.mark.parametrize('condition', ['hover', '115kt', '160kt'])
def test_table_rows_add_up(condition):
    """Each printed row: hub couple plus rotor force equals the total."""
    force, total = TABLE[condition]
    p = _run_flapping_moment_buildup(dCHsigma_da1s=_dCH_for(force))

    assert_near_equal(p.get_val('dMCG_da1s_force')[0], force, 1e-10)
    assert_near_equal(p.get_val('dMCG_da1s')[0], total, 1e-10)
    assert_near_equal(K_HUB_flapping_moment_buildup + force, total, 1e-12)


def test_hover_entry_from_first_principles():
    """The only row checkable without a full trim, since lambda' = -sqrt(C_T/2).

    Gives 69,400 against the 73,500 printed, 5.6 % low. The table's theta_75
    most likely comes from the Chapter 1 hover analysis rather than from the
    idealised momentum value used here.
    """
    lam = -np.sqrt(CT_flapping_moment_buildup / 2.0)
    dCH = CT_SIGMA_flapping_moment_buildup + A_EX_flapping_moment_buildup / 8 * lam

    p = _run_flapping_moment_buildup(dCHsigma_da1s=dCH)
    assert_near_equal(p.get_val('dMCG_da1s_force')[0], 69414.0, 1e-3)
    assert_near_equal(p.get_val('dMCG_da1s_force')[0], TABLE['hover'][0], 6e-2)


def test_the_rotor_force_contribution_peaks_around_115_knots():
    """Two effects compete in lambda': the induced velocity collapsing with
    speed, and the tip path plane pitching further nose down.
    """
    hover, kt115, kt160 = (TABLE[c][0] for c in ('hover', '115kt', '160kt'))
    assert hover < kt115
    assert kt160 < kt115


def test_implied_tip_path_plane_angles_are_plausible():
    """Working back from the table gives sensible attitudes at each speed."""
    for condition, speed_kt in (('115kt', 115.0), ('160kt', 160.0)):
        mu = speed_kt * 1.6878098 / OMEGA_R_flapping_moment_buildup
        lam = (_dCH_for(TABLE[condition][0]) - CT_SIGMA_flapping_moment_buildup) / (A_EX_flapping_moment_buildup / 8)
        v1 = CT_flapping_moment_buildup / (2 * mu)
        alpha_tpp = np.degrees((lam + v1) / mu)

        assert -5.0 < alpha_tpp < 0.0

    # and it gets steeper with speed
    def tpp(condition, speed_kt):
        mu = speed_kt * 1.6878098 / OMEGA_R_flapping_moment_buildup
        lam = (_dCH_for(TABLE[condition][0]) - CT_SIGMA_flapping_moment_buildup) / (A_EX_flapping_moment_buildup / 8)
        return (lam + CT_flapping_moment_buildup / (2 * mu)) / mu

    assert tpp('160kt', 160.0) < tpp('115kt', 115.0)


def test_hub_couple_does_not_vary_with_flight_condition():
    """p. 479 repeats 200,940 on all three rows."""
    nn = 3
    p = _run_flapping_moment_buildup(nn, dCHsigma_da1s=np.array([_dCH_for(TABLE[c][0])
                                         for c in ('hover', '115kt', '160kt')]))
    totals = p.get_val('dMCG_da1s') - p.get_val('dMCG_da1s_force')
    assert_near_equal(totals, np.full(nn, K_HUB_flapping_moment_buildup), 1e-10)


def test_roll_axis_is_the_symmetric_analogue():
    dCH = 0.0384
    p = _run_flapping_moment_buildup(dCHsigma_da1s=dCH, dCYsigma_db1s=dCH)
    assert_near_equal(p.get_val('dLCG_db1s')[0], p.get_val('dMCG_da1s')[0],
                      1e-14)


def test_scale_is_the_thrust_normaliser():
    """sigma rho pi R^4 Omega^2 times C_T/sigma must return the thrust."""
    assert_near_equal(CT_SIGMA_flapping_moment_buildup * SCALE_flapping_moment_buildup, GW_flapping_moment_buildup, 1e-10)


def test_zero_arm_leaves_only_the_hub_couple():
    p = _run_flapping_moment_buildup(h_M=0.0, dCHsigma_da1s=0.0384)
    assert_near_equal(p.get_val('dMCG_da1s_force')[0], 0.0, 1e-14)
    assert_near_equal(p.get_val('dMCG_da1s')[0], K_HUB_flapping_moment_buildup, 1e-13)


def test_vectorized_flapping_moment_buildup():
    nn = 3
    dCH = np.array([0.0384, 0.0679, 0.0596])
    p = _run_flapping_moment_buildup(nn, dCHsigma_da1s=dCH)
    assert_near_equal(p.get_val('dMCG_da1s_force'), dCH * SCALE_flapping_moment_buildup * H_M_flapping_moment_buildup, 1e-12)



@pytest.mark.parametrize('nn', [1, 4])
def test_partials_flapping_moment_buildup(nn):
    p = _run_flapping_moment_buildup(nn, dMM_da1s=np.linspace(1.8e5, 2.2e5, nn),
             dCHsigma_da1s=np.linspace(0.03, 0.07, nn),
             dCYsigma_db1s=np.linspace(0.03, 0.07, nn),
             rho=np.linspace(0.0018, 0.00238, nn),
             Omega=np.linspace(19.0, 21.7, nn))
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-6, rtol=1e-10)


def test_chained_from_stiffness_and_h_force():
    """FlappingMomentsGroup and HForceFlappingDerivComp feeding the build-up."""
    from prouty.flapping import (FlappingMomentsGroup, HForceFlappingDerivComp)

    p = om.Problem()
    m = p.model
    m.add_subsystem('moments', FlappingMomentsGroup(convention='center'),
                    promotes=['*'])
    m.add_subsystem('hforce', HForceFlappingDerivComp(), promotes=['*'])
    m.add_subsystem('buildup', FlappingMomentBuildupComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)

    I_b_from_gamma = 2.0 * RHO_SL_flapping_moment_buildup * A_EX_flapping_moment_buildup * R_EX_flapping_moment_buildup ** 4 / 8.1
    p.set_val('R', R_EX_flapping_moment_buildup)
    p.set_val('e_over_R', 0.05)
    p.set_val('I_b_ref', I_b_from_gamma)
    p.set_val('b', 4.0)
    p.set_val('Omega', OMEGA_EX_flapping_moment_buildup)
    p.set_val('sigma', SIGMA_flapping_moment_buildup)
    p.set_val('rho', RHO_SL_flapping_moment_buildup)
    p.set_val('h_M', H_M_flapping_moment_buildup)
    p.set_val('CT_sigma', CT_SIGMA_flapping_moment_buildup)
    p.set_val('mu', 0.0)
    p.set_val('alpha_s', 0.0)
    p.set_val('a_1s', 0.0)
    p.set_val('v1_over_OmegaR', np.sqrt(CT_flapping_moment_buildup / 2.0))
    p.set_val('T', GW_flapping_moment_buildup)
    p.run_model()

    assert_near_equal(p.get_val('dMM_da1s')[0], K_HUB_flapping_moment_buildup, 1e-4)
    assert_near_equal(p.get_val('dMCG_da1s')[0], 274440.0, 2e-2)

    data = p.check_totals(of=['dMCG_da1s'],
                          wrt=['R', 'e_over_R', 'b', 'Omega', 'sigma', 'rho',
                               'h_M', 'CT_sigma', 'v1_over_OmegaR'],
                          method='cs', compact_print=True, out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-8, atol=1e-4)


# ------------------------------------------------------------------------
# h_force_flapping_group
# ------------------------------------------------------------------------

GW_h_force_flapping_grp, R_EX_h_force_flapping_grp, OMEGA_R_h_force_flapping_grp, SIGMA_h_force_flapping_grp, A_EX_h_force_flapping_grp, H_M_h_force_flapping_grp = 20000.0, 30.0, 650.0, 0.085, 6.0, 7.5


RHO_SL_h_force_flapping_grp, OMEGA_EX_h_force_flapping_grp = 0.002378, 650.0 / 30.0


SCALE_h_force_flapping_grp = SIGMA_h_force_flapping_grp * RHO_SL_h_force_flapping_grp * np.pi * R_EX_h_force_flapping_grp ** 4 * OMEGA_EX_h_force_flapping_grp ** 2


CT_h_force_flapping_grp = GW_h_force_flapping_grp / (RHO_SL_h_force_flapping_grp * np.pi * R_EX_h_force_flapping_grp ** 2 * OMEGA_R_h_force_flapping_grp ** 2)


CT_SIGMA_h_force_flapping_grp = CT_h_force_flapping_grp / SIGMA_h_force_flapping_grp


K_HUB_h_force_flapping_grp = 200940.0


def _build(nn=1, **opts):
    over = {k: opts.pop(k) for k in list(opts) if k in
            ('mu', 'alpha_s', 'a_1s', 'v1_over_OmegaR', 'theta_75',
             'dMM_da1s', 'CT_sigma')}

    p = om.Problem()
    p.model.add_subsystem('g5', HForceFlappingGroup(num_nodes=nn, **opts),
                          promotes=['*'])
    p.setup(force_alloc_complex=True)

    p.set_val('CT_sigma', np.full(nn, CT_SIGMA_h_force_flapping_grp))
    p.set_val('a', np.full(nn, A_EX_h_force_flapping_grp))
    if opts.get('include_buildup', True):
        p.set_val('R', R_EX_h_force_flapping_grp)
        p.set_val('sigma', SIGMA_h_force_flapping_grp)
        p.set_val('h_M', H_M_h_force_flapping_grp)
        p.set_val('rho', np.full(nn, RHO_SL_h_force_flapping_grp))
        p.set_val('Omega', np.full(nn, OMEGA_EX_h_force_flapping_grp))
        p.set_val('dMM_da1s', np.full(nn, K_HUB_h_force_flapping_grp))
    for k, v in over.items():
        p.set_val(k, v)

    p.run_model()
    return p


def _hover():
    return _build(mu=0.0, alpha_s=0.0, a_1s=0.0,
                  v1_over_OmegaR=np.sqrt(CT_h_force_flapping_grp / 2.0))


def test_hover_reproduces_the_page_479_row():
    """73,500 rotor force and 274,440 total, to within the 5.6 % of C7-11."""
    p = _hover()

    assert_near_equal(p.get_val('dMCG_da1s_force')[0], 73500.0, 6e-2)
    assert_near_equal(p.get_val('dMCG_da1s')[0], 274440.0, 2e-2)


def test_inflow_more_than_halves_the_in_plane_force():
    p = _hover()
    force = p.get_val('dCHsigma_da1s')[0] * SCALE_h_force_flapping_grp
    assert_near_equal(GW_h_force_flapping_grp / force, 2.16, 1e-2)


def test_both_forms_agree_in_hover():
    lam = -np.sqrt(CT_h_force_flapping_grp / 2.0)
    th75 = 1.5 * (4 * CT_SIGMA_h_force_flapping_grp / A_EX_h_force_flapping_grp - lam)

    by_lambda = _hover()
    by_theta = _build(form='theta75', theta_75=th75)

    assert_near_equal(by_theta.get_val('dCHsigma_da1s')[0],
                      by_lambda.get_val('dCHsigma_da1s')[0], 1e-12)
    assert_near_equal(by_theta.get_val('dMCG_da1s')[0],
                      by_lambda.get_val('dMCG_da1s')[0], 1e-10)


def test_roll_axis_tracks_the_pitch_axis():
    p = _hover()
    assert_near_equal(p.get_val('dCYsigma_db1s')[0],
                      p.get_val('dCHsigma_da1s')[0], 1e-14)
    assert_near_equal(p.get_val('dLCG_db1s')[0], p.get_val('dMCG_da1s')[0],
                      1e-13)


def test_buildup_can_be_switched_off():
    p = _build(include_buildup=False, mu=0.0, alpha_s=0.0, a_1s=0.0,
               v1_over_OmegaR=np.sqrt(CT_h_force_flapping_grp / 2.0))
    names = {meta['prom_name'] for _, meta in
             p.model.list_outputs(prom_name=True, val=False, out_stream=None)}
    assert 'dMCG_da1s' not in names
    assert 'dCHsigma_da1s' in names
    assert p.get_val('dCHsigma_da1s')[0] > 0.0


def test_unconnected_stiffness_leaves_only_the_force_term():
    p = _build(mu=0.0, alpha_s=0.0, a_1s=0.0,
               v1_over_OmegaR=np.sqrt(CT_h_force_flapping_grp / 2.0), dMM_da1s=0.0)
    assert_near_equal(p.get_val('dMCG_da1s')[0],
                      p.get_val('dMCG_da1s_force')[0], 1e-14)


def test_derivative_falls_with_a_more_negative_lambda_prime():
    nn = 4
    alpha = np.radians(np.array([0.0, -3.0, -6.0, -12.0]))
    p = _build(nn, mu=np.full(nn, 0.3), alpha_s=alpha, a_1s=np.zeros(nn),
               v1_over_OmegaR=np.full(nn, 0.0118))

    assert np.all(np.diff(p.get_val('lambda_prime')) < 0)
    assert np.all(np.diff(p.get_val('dCHsigma_da1s')) < 0)
    assert np.all(np.diff(p.get_val('dMCG_da1s')) < 0)


def test_group_is_feed_forward():
    p = _hover()
    before = p.get_val('dMCG_da1s').copy()
    p.run_model()
    assert_near_equal(p.get_val('dMCG_da1s'), before, 1e-14)



@pytest.mark.parametrize('form', ['lambda', 'theta75'])
def test_totals(form):
    nn = 3
    extra = ({'theta_75': np.linspace(0.10, 0.20, nn)} if form == 'theta75'
             else {'mu': np.linspace(0.1, 0.45, nn),
                   'alpha_s': np.radians(np.linspace(-8.0, -2.0, nn)),
                   'a_1s': np.radians(np.linspace(0.0, 4.0, nn)),
                   'v1_over_OmegaR': np.linspace(0.008, 0.06, nn)})
    p = _build(nn, form=form, CT_sigma=np.linspace(0.06, 0.09, nn),
               dMM_da1s=np.linspace(1.8e5, 2.2e5, nn), **extra)

    wrt = ['CT_sigma', 'a', 'R', 'sigma', 'h_M', 'rho', 'Omega', 'dMM_da1s']
    wrt += ['theta_75'] if form == 'theta75' else ['mu', 'alpha_s', 'a_1s',
                                                   'v1_over_OmegaR']

    data = p.check_totals(of=['dCHsigma_da1s', 'dMCG_da1s', 'dLCG_db1s'],
                          wrt=wrt, method='cs', compact_print=True,
                          out_stream=None)
    for val in data.values():
        analytic = val['J_fwd'] if 'J_fwd' in val else val['J_rev']
        np.testing.assert_allclose(analytic, val['J_fd'], rtol=1e-8, atol=1e-4)


def test_full_chapter_chain():
    """G2, G4 and G5 together, with the p. 479 correction fed back into the
    c.g. moment through CGMomentComp(inplane_force='external').
    """
    from prouty.flapping import (CGMomentComp, ForwardFlightFlappingGroup,
                                 FlappingMomentsGroup)

    nn = 2
    p = om.Problem()
    m = p.model
    m.add_subsystem('steady', ForwardFlightFlappingGroup(num_nodes=nn),
                    promotes=['*'])
    m.add_subsystem('moments',
                    FlappingMomentsGroup(num_nodes=nn, blade_input='external',
                                         inplane_force='external'),
                    promotes=['*'])
    m.add_subsystem('hforce', HForceFlappingGroup(num_nodes=nn),
                    promotes=['*'])
    p.setup()

    p.set_val('e_over_R', 0.05)
    p.set_val('R', R_EX_h_force_flapping_grp)
    p.set_val('I_b_ref', 2870.0)
    p.set_val('sigma', SIGMA_h_force_flapping_grp)
    p.set_val('b', 4.0)
    p.set_val('a', np.full(nn, A_EX_h_force_flapping_grp))
    p.set_val('gamma', np.full(nn, 8.1))
    p.set_val('rho', np.full(nn, RHO_SL_h_force_flapping_grp))
    p.set_val('Omega', np.full(nn, OMEGA_EX_h_force_flapping_grp))
    p.set_val('mu', np.array([0.2, 0.4]))
    p.set_val('theta_0', np.full(nn, np.radians(14.0)))
    p.set_val('theta_1', np.full(nn, np.radians(-10.0)))
    p.set_val('alpha_s', np.full(nn, np.radians(-5.0)))
    p.set_val('A_1', np.full(nn, np.radians(-2.2)))
    p.set_val('B_1', np.full(nn, np.radians(1.5)))
    p.set_val('T', np.full(nn, GW_h_force_flapping_grp))
    p.set_val('h_M', H_M_h_force_flapping_grp)
    p.run_model()

    # CGMomentComp needs the dimensional in-plane force, not the derivative
    assert np.all(p.get_val('lambda_prime') < 0.0)
    assert np.all(p.get_val('dMCG_da1s') > p.get_val('dMM_da1s'))
