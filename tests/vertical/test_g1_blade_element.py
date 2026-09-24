"""Chapter 2, G1 -- blade element in vertical flight: HoverRotorGroup(flight='climb'), pp. 95-97."""

import warnings

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import (assert_check_partials, assert_check_totals,
                                         assert_near_equal)

from prouty.hover import ClimbInflowRatioComp, HoverRotorGroup

RHO_SL, A_M, GW = 0.002377, np.pi * 30.0 ** 2, 20000.0
V_HOV = np.sqrt(GW / (2 * RHO_SL * A_M))


def _rotor(mode='trim', flight='climb'):
    p = om.Problem()
    p.model.add_subsystem('r', HoverRotorGroup(mode=mode, flight=flight), promotes=['*'])
    p.setup(force_alloc_complex=True)
    if mode == 'trim':
        p.set_val('T_target', GW)
    else:
        p.set_val('theta_0', 17.5)
    return p


def _run(p, V_c=None):
    if V_c is not None:
        p.set_val('V_c', V_c, units='ft/s')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p.run_model()
    return p


@pytest.fixture(scope='module')
def hover_trim():
    p = _run(_rotor(flight='hover'))
    return p.get_val('theta_0')[0], p.get_val('power_hp')[0]


# ------------------------------------------------------------------------
# the equation of p. 96
# ------------------------------------------------------------------------

def test_inflow_equals_p96_dimensional_form():
    """v1 = [-(Omega a c b/2 + 4 pi V_c) + sqrt((...)^2 + 8 pi b Omega^2 a c r (theta - V_c/Omega r))] / 8 pi."""
    r_R, c_R, theta_deg, a_deg = np.array([0.3, 0.6, 0.9]), np.full(3, 2 / 30), \
        np.array([14.0, 11.0, 8.0]), np.full(3, 0.105)
    R, b, V_tip, V_c = 30.0, 4.0, 650.0, 12.0
    p = om.Problem()
    p.model.add_subsystem('c', ClimbInflowRatioComp(num_nodes=3), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in dict(r_R=r_R, c_R=c_R, theta=theta_deg, a=a_deg, b=b, V_tip=V_tip, V_c=V_c).items():
        p.set_val(k, v)
    p.run_model()

    Om, r, c, a, th = V_tip / R, r_R * R, c_R * R, np.degrees(a_deg), np.radians(theta_deg)
    B = Om * a * c * b / 2 + 4 * np.pi * V_c
    v1 = (-B + np.sqrt(B ** 2 + 8 * np.pi * b * Om ** 2 * a * c * r * (th - V_c / (Om * r)))) \
        / (8 * np.pi)
    assert_near_equal(p.get_val('vi_Or'), v1 / (Om * r), 1e-12)
    assert_near_equal(p.get_val('v1_Or'), (V_c + v1) / (Om * r), 1e-12)

    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-12, rtol=1e-10)


def test_hover_is_chapter1_exactly():
    """flight='climb' at V_c = 0 reproduces the Chapter 1 procedure (Figure 1.45 case)."""
    climb, hover = _run(_rotor('analysis'), 0.0), _run(_rotor('analysis', 'hover'))
    for name in ('T', 'power_hp', 'FM', 'alpha', 'v1_Or'):
        assert_near_equal(climb.get_val(name), hover.get_val(name), 1e-12)


# ------------------------------------------------------------------------
# statements of pp. 95-97
# ------------------------------------------------------------------------

def test_climb_needs_more_pitch_at_the_same_angle_of_attack(hover_trim):
    """p. 96: same thrust, the mean angle of attack barely moves, the pitch goes up."""
    p0 = _run(_rotor(), 0.0)
    a0 = p0.get_val('alpha')[p0.get_val('r_R') > 0.5].mean()
    p = _run(_rotor(), 8.33)
    a1 = p.get_val('alpha')[p.get_val('r_R') > 0.5].mean()
    d_theta = p.get_val('theta_0')[0] - hover_trim[0]
    assert abs(a1 - a0) < 0.1 * d_theta
    assert d_theta > 0.5


def test_collective_increment_against_g3_and_c2_1(hover_trim):
    """500 ft/min: the blade element gives 0.55 deg, the p. 101 equation 0.52, the printed 0.4 (C2-1)."""
    d_theta = _run(_rotor(), 8.33).get_val('theta_0')[0] - hover_trim[0]
    v_sum = 8.33 / 2 + np.sqrt(8.33 ** 2 / 4 + V_HOV ** 2)
    g3 = np.degrees((v_sum - V_HOV) / (0.75 * 650.0))
    assert 1.0 < d_theta / g3 < 1.12


@pytest.mark.parametrize('V_c', [8.33, 16.67, 33.3])
def test_climb_power_against_momentum(V_c, hover_trim):
    """The blade element climb power is 10-12 % above G.W.(v_1c + V_c - v_1hov)/550 (C2-11)."""
    dP = _run(_rotor(), V_c).get_val('power_hp')[0] - hover_trim[1]
    v_sum = V_c / 2 + np.sqrt(V_c ** 2 / 4 + V_HOV ** 2)
    assert 1.09 < dP / (GW * (v_sum - V_HOV) / 550.0) < 1.13


def test_low_descent_reduces_pitch_and_power(hover_trim):
    p = _run(_rotor(), -5.0)
    assert p.get_val('theta_0')[0] < hover_trim[0]
    assert p.get_val('power_hp')[0] < hover_trim[1]


def test_radicand_positive_for_any_rate_with_positive_pitch():
    """(lam - 2P)^2 + 8 P theta >= 0: no warning, even far outside the momentum range."""
    p = _rotor('analysis')
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        warnings.filterwarnings('ignore', message='.*Figure 1.34.*')
        for V_c in (-3.0 * V_HOV, 3.0 * V_HOV):
            p.set_val('V_c', V_c)
            p.run_model()
    assert np.all(np.isfinite(p.get_val('v1_Or')))


def test_totals_analysis_mode():
    p = _run(_rotor('analysis'), 10.0)
    data = p.check_totals(of=['T', 'power_hp'], wrt=['V_c', 'theta_0'], method='cs',
                          out_stream=None)
    assert_check_totals(data, atol=1e-7, rtol=1e-7)
