"""Chapter 5, G3 -- maximum acceleration (pp. 364-365); G4 -- maximum deceleration (pp. 365-366)."""
import importlib.util
import pathlib

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.special_performance import (HoverAccelerationComp, AvailableTorqueComp,
                                        SmoothMinComp, MaxAccelerationGroup, WeightCoefComp,
                                        DecelerationForceComp, RotorForceLimitGroup,
                                        MaxDecelerationGroup)

# Figures 5.14 and 5.15 (pp. 365-366), digitized: speed [kt] -> ft/s^2
FIG_5_14 = {40: 22.9, 60: 18.1, 80: 14.0, 100: 10.2, 120: 6.6, 140: 3.3, 160: 0.0}
FIG_5_15 = {40: 17.8, 60: 12.4, 80: 8.7, 100: 6.4, 120: 5.7, 140: 5.7, 160: 6.4}
SPEEDS = np.array([40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0])
ROTOR = dict(sigma=0.0849, R=30.0)


def _run(system, **inputs):
    p = om.Problem()
    p.model.add_subsystem('s', system, promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in inputs.items():
        p.set_val(k, v[0], units=v[1]) if isinstance(v, tuple) else p.set_val(k, v)
    p.run_model()
    return p


# ---------------- G3 ----------------

def test_anchor_fig514_hover_acceleration():
    """T_max = 27,800 lb (Figure 4.35) at 20,000 lb: 31 ft/s^2 at zero speed on Fig. 5.14."""
    assert_near_equal(_run(HoverAccelerationComp()).get_val('acc_hover'), 31.0, 0.01)


@pytest.fixture(scope='module')
def accel():
    V = np.concatenate(([20.0], SPEEDS))
    n = len(V)
    return _run(MaxAccelerationGroup(num_nodes=n), V=(V, 'kn'), cd_bar=0.01 * np.ones(n),
                **ROTOR)


def test_accel_balance(accel):
    """Rotor torque at the available main rotor power (3,600 hp), vertical thrust = G.W."""
    assert np.max(np.abs(accel.get_val('CQ_sigma') - accel.get_val('CQ_sigma_avail'))) < 1e-10
    assert np.all(accel.get_val('alpha_TPP') < 0.0)                 # tilted forward


def test_anchor_fig514_forward_acceleration(accel):
    """20-80 kt within 15 % of Fig. 5.14 (chain above it); the gap grows with speed
    (+25 % at 100 kt, 4.9 ft/s^2 left at 160 kt) because the closed-form rotor has
    no stall torque at C_T/sigma = 0.09-0.10 (C5-6)."""
    acc = accel.get_val('acc_max')[1:]
    fig = np.array([FIG_5_14[v] for v in SPEEDS])
    low = SPEEDS <= 80
    assert np.all(np.abs(acc[low] / fig[low] - 1) < 0.15)
    assert np.all(acc[~low] > fig[~low])
    assert np.all(np.diff(accel.get_val('acc_max')) < 0)
    assert_near_equal(accel.get_val('acc_max')[0], 28.4, 0.08)      # 20 kt


def test_available_power_is_a_free_input():
    """P_MR_avail: default 3,600 hp, promoted to the group, drives the capability."""
    from prouty.special_performance.available_torque_comp import DEFAULT_P_MR_AVAIL
    base = _run(MaxAccelerationGroup(), V=(100.0, 'kn'), cd_bar=0.01, **ROTOR)
    assert base.get_val('P_MR_avail', units='hp')[0] == DEFAULT_P_MR_AVAIL == 3600.0
    low = _run(MaxAccelerationGroup(), V=(100.0, 'kn'), cd_bar=0.01, P_MR_avail=3000.0, **ROTOR)
    assert low.get_val('acc_max')[0] < base.get_val('acc_max')[0]


def test_acceleration_is_hover_limited_at_low_speed(accel):
    assert accel.get_val('acc_max')[0] <= accel.get_val('acc_hover')[0]


# ---------------- G4 ----------------

@pytest.fixture(scope='module')
def decel():
    n = len(SPEEDS)
    return _run(RotorForceLimitGroup(num_nodes=n, mode='decel'), V=(SPEEDS, 'kn'),
                cd_bar=0.01 * np.ones(n), **ROTOR)


def test_autorotation_balance(decel):
    assert np.max(np.abs(decel.get_val('CQ_sigma'))) < 1e-10
    cw = 20000.0 / (0.002377 * 240.0 * 780.0 ** 2)
    assert_near_equal(decel.get_val('CT_sigma') * np.cos(decel.get_val('alpha_TPP')),
                      cw * np.ones(len(SPEEDS)), 1e-10)


def test_anchor_fig515_deceleration(decel):
    """120 % rotor speed: within 12 % of Fig. 5.15 from 60 to 160 kt, including the
    minimum near 120-140 kt; 20 % high at 40 kt."""
    d = decel.get_val('decel')
    fig = np.array([FIG_5_15[v] for v in SPEEDS])
    assert np.all(np.abs(d[1:] / fig[1:] - 1) < 0.12)
    assert np.argmin(d) in (4, 5)


def test_autorotation_limit_starts_near_37_kt():
    """Below ~35 kt the rotor cannot autorotate at 120 % speed with a flare under 45 deg
    (37.6 deg at 37 kt, 59 deg at 30 kt): the autorotation branch exists above
    37 kt, as the note of Fig. 5.15 says; below, the acceleration capability rules."""
    p = _run(RotorForceLimitGroup(mode='decel'), V=(37.0, 'kn'), cd_bar=0.01, **ROTOR)
    assert_near_equal(p.get_val('alpha_TPP', units='deg'), 37.6, 0.02)
    p = _run(MaxDecelerationGroup(num_nodes=2), V=(np.array([37.0, 60.0]), 'kn'),
             acc_max=[24.0, 19.9], cd_bar=0.01 * np.ones(2), **ROTOR)
    lim, auto = p.get_val('decel_max'), p.get_val('decel')
    assert lim[0] < min(24.0, auto[0])
    assert lim[1] == pytest.approx(auto[1], abs=0.3)            # autorotation-limited


# ---------------- derivatives ----------------

@pytest.mark.parametrize('comp, inputs', [
    (HoverAccelerationComp(), {}),
    (AvailableTorqueComp(), {}),
    (SmoothMinComp(num_nodes=3, a_scalar=True), {'acc_hover': 31.0,
                                                 'acc_power': [40.0, 31.2, 10.0]}),
    (WeightCoefComp(num_nodes=2), {'V': [100.0, 200.0]}),
    (DecelerationForceComp(num_nodes=2), {'alpha_TPP': [0.1, 0.4], 'CH_sigma': [0.001, 0.0]}),
    (DecelerationForceComp(num_nodes=2, output='acc'), {'alpha_TPP': [-0.4, -0.5]}),
])
def test_partials(comp, inputs):
    p = _run(comp, **inputs)
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-9, rtol=1e-9)


def test_accel_totals_through_the_balance():
    p = _run(MaxAccelerationGroup(num_nodes=2), V=(np.array([60.0, 120.0]), 'kn'),
             cd_bar=0.01 * np.ones(2), **ROTOR)
    from openmdao.utils.assert_utils import assert_check_totals
    assert_check_totals(p.check_totals(of=['acc_max'], wrt=['GW', 'P_MR_avail', 'f', 'T_max'],
                                       method='cs', out_stream=None), atol=1e-6, rtol=1e-6)


def test_decel_totals_through_the_balance():
    p = _run(RotorForceLimitGroup(num_nodes=2, mode='decel'), V=(np.array([80.0, 120.0]), 'kn'),
             cd_bar=0.01 * np.ones(2), **ROTOR)
    from openmdao.utils.assert_utils import assert_check_totals
    assert_check_totals(p.check_totals(of=['decel'], wrt=['GW', 'V_tip', 'f', 'cd_bar'],
                                       method='cs', out_stream=None), atol=1e-6, rtol=1e-6)
