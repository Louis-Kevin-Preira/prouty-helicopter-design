"""Chapter 5, G2e -- minimum touchdown speed, pp. 358-363."""
import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import (assert_check_partials, assert_check_totals,
                                         assert_near_equal)

from prouty.special_performance.book_figures import FIG_5_12
from prouty.special_performance import (FlarePitchRateComp, FlareTimeComp, FlareAngleComp,
                                        FlareConditionsComp, FlareAutorotationGroup,
                                        TouchdownSpeedComp, MinTouchdownSpeedGroup)

ROTOR = dict(sigma=0.0849, cd_bar=0.01)          # example helicopter, theta_1 = -10 deg
# The two printed times, 0.8 s (p. 363) and 1.25 s (p. 362), are met together by
# (C_T/sigma)_max = 0.1406 and hp_OGE = 1,643 hp with J = 11,735 slug ft^2.
FLARE = dict(CW_sigma=0.083, CT_sigma_max=0.1406, P_OGE=1643.0, **ROTOR)


def _run(system, **inputs):
    p = om.Problem()
    p.model.add_subsystem('s', system, promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in inputs.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_anchor_p362_pitch_rate():
    """88 deg/s with Delta_B1 = 8 deg, gamma = 8.1, Omega = 21.67 (p. 362)."""
    p = _run(FlarePitchRateComp())
    assert_near_equal(p.get_val('theta_dot_max', units='deg/s'), 88.0, 0.005)


def test_anchor_p362_flare_time_and_angle():
    """Delta_t = 1.25 s; alpha = 110 deg (printed 100, C5-8) -> 45 deg (p. 362)."""
    p = _run(MinTouchdownSpeedGroup(), **FLARE)
    assert_near_equal(p.get_val('dt'), 1.25, 0.005)
    assert_near_equal(p.get_val('alpha_raw', units='deg'), 110.0, 0.01)
    assert_near_equal(p.get_val('alpha_TPP', units='deg'), 45.0, 1e-3)


def test_flare_time_consistent_with_equivalent_hover_time():
    """Same J, Omega, hover power: t_equiv of p. 363 is Delta_t with 0.8 (C_T/sigma)_max."""
    from prouty.special_performance import EquivalentHoverTimeComp
    te = _run(EquivalentHoverTimeComp(), CW_sigma=0.083, CT_sigma_max=0.1406, P_OGE=1643.0)
    assert_near_equal(te.get_val('t_equiv'), 0.8, 0.01)


@pytest.mark.parametrize('cw', sorted(FIG_5_12))
def test_mu_auto_against_fig512(cw):
    """Closed-form rotor (theta_1 = -10 deg) against Figure 5.12 (charts, -5 deg).

    C_W/sigma = 0.05: within 10 % from 10 to 45 deg. C_W/sigma = 0.10: within
    10 % from 30 deg; at 15-20 deg (C_T/sigma = 0.104-0.106) the charts are in
    retreating blade stall and the closed form is 16-25 % low (see C5-6)."""
    ang = np.array(sorted(FIG_5_12[cw]))
    n = len(ang)
    p = _run(FlareAutorotationGroup(num_nodes=n), alpha_TPP=np.radians(ang),
             CW_sigma=cw * np.ones(n), sigma=0.0849, cd_bar=0.01 * np.ones(n))
    mu = p.get_val('mu_auto')
    fig = np.array([FIG_5_12[cw][a] for a in ang])
    ok = (ang >= 30) | (cw < 0.07)
    assert np.all(np.abs(mu[ok] / fig[ok] - 1) < 0.10)
    assert np.all(np.diff(mu) < 0)


def test_anchor_p362_touchdown_speed():
    """Printed 21 kt; the chain gives 18.9 kt (mu_auto 0.080 against 0.082 on Fig. 5.12)."""
    p = _run(MinTouchdownSpeedGroup(), **FLARE)
    assert_near_equal(p.get_val('V_TD', units='kn'), 21.0, 0.12)


def test_touchdown_speed_with_figure_mu():
    """With mu_auto read on Figure 5.12 (0.082 at 45 deg, C_W/sigma = 0.083)."""
    p = _run(TouchdownSpeedComp(), mu_auto=0.082, dt=1.25)
    assert_near_equal(p.get_val('V_TD', units='kn'), (0.082 * 650 - 16.1 * 1.25) / 1.6878, 1e-4)


def test_autorotation_residual_is_zero():
    p = _run(FlareAutorotationGroup(num_nodes=2), alpha_TPP=np.radians([25.0, 40.0]),
             CW_sigma=[0.07, 0.09], sigma=0.0849, cd_bar=[0.01, 0.01])
    assert np.all(np.abs(p.get_val('CQ_sigma')) < 1e-10)
    assert_near_equal(p.get_val('CT_sigma'), np.array([0.07, 0.09]) / np.cos(np.radians([25, 40])),
                      1e-12)


@pytest.mark.parametrize('comp, inputs', [
    (FlarePitchRateComp(num_nodes=2), {'delta_B1': np.radians([6.0, 8.0])}),
    (FlareTimeComp(num_nodes=2), {'CT_sigma_max': [0.14, 0.15]}),
    (FlareAngleComp(num_nodes=3), {'theta_dot_max': [0.5, 0.7, 1.5], 'dt': [1.0, 1.1, 1.2]}),
    (FlareConditionsComp(num_nodes=2), {'alpha_TPP': [0.3, 0.7]}),
    (TouchdownSpeedComp(num_nodes=2), {'alpha_TPP': [0.5, 0.78]}),
])
def test_partials(comp, inputs):
    p = _run(comp, **inputs)
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-9, rtol=1e-9)


def test_totals_through_the_autorotation_balance():
    p = _run(MinTouchdownSpeedGroup(), **{**FLARE, 'delta_B1': np.radians(2.0)})
    assert p.get_val('alpha_TPP', units='deg') < 30.0          # off the 45 deg limit
    assert_check_totals(p.check_totals(of=['V_TD'], wrt=['CW_sigma', 'delta_B1', 'P_OGE',
                                                          'cd_bar', 'CT_sigma_max'],
                                       method='cs', out_stream=None), atol=1e-6, rtol=1e-6)
