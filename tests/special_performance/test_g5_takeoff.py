"""Chapter 5, G5 -- takeoff at high gross weight, pp. 366-368 (formula layer)."""
import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.special_performance import (TakeoffAccelerationDistanceComp, ClimboutDistanceComp,
                                        TakeoffDistanceGroup)


def _run(system, **inputs):
    p = om.Problem()
    p.model.add_subsystem('s', system, promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in inputs.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_acceleration_distance_integrates_the_linear_law():
    """x_acc and t_acc of p. 367 against a numerical integration of x_ddot = a0 (1 - x_dot/V_max)."""
    a0, Vm = 12.0, 240.0
    V = np.array([20.0, 45.0, 80.0])
    p = _run(TakeoffAccelerationDistanceComp(num_nodes=3), acc_0=a0, V_max=Vm, V_rot=V)
    dt = 1e-4
    t, x, v, got = 0.0, 0.0, 0.0, []
    for target in V:
        while v < target:
            a = a0 * (1 - v / Vm)
            x += v * dt + 0.5 * a * dt ** 2
            v += a * dt
            t += dt
        got.append((x, t))
    got = np.array(got)
    assert_near_equal(p.get_val('x_acc'), got[:, 0], 2e-3)
    assert_near_equal(p.get_val('t_acc'), got[:, 1], 2e-3)


def test_climbout_is_the_climb_path():
    """x_CL = h / tan(gamma) with tan(gamma) = (R/C)/V (p. 368)."""
    p = _run(ClimboutDistanceComp(num_nodes=2), P_avail=4000.0, P_level=[3300.0, 3400.0],
             GW=28000.0, h=50.0, V_rot=[40.0, 50.0])
    rc = 550.0 * np.array([700.0, 600.0]) / 28000.0
    assert_near_equal(p.get_val('R_C'), rc, 1e-12)
    assert_near_equal(p.get_val('x_cl'), 50.0 * np.array([40.0, 50.0]) / rc, 1e-12)
    assert_near_equal(p.get_val('R_C', units='ft/min'), 33000.0 * np.array([700.0, 600.0]) / 28000.0,
                      1e-9)


@pytest.mark.parametrize('comp, inputs', [
    (TakeoffAccelerationDistanceComp(num_nodes=3), {'V_rot': [10.0, 60.0, 150.0]}),
    (ClimboutDistanceComp(num_nodes=2), {'x_acc': [100.0, 200.0]}),
    (TakeoffDistanceGroup(num_nodes=2), {'V_rot': [30.0, 60.0]}),
])
def test_partials(comp, inputs):
    p = _run(comp, **inputs)
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ---------------- low-speed power and optimum rotation speed ----------------

import importlib.util
import pathlib

from prouty.special_performance import (LowSpeedPowerComp, JoinStencilComp, JoinSlopeComp,
                                        LowSpeedPowerGroup, TakeoffStencilComp,
                                        OptimumTakeoffGroup)

KT = 1.6878
# Example at 28,000 lb, sea level (Figure 5.16), from the Chapter 4 hover chain:
P_HOVER_OGE = 4209.0      # hp, HoverPerformanceGroup P_req (OGE)
P_AVAIL = 4077.0          # hp, installed takeoff power (C4-27)
T_MAX_IGE = 31700.0       # lb, IGE capability at Z/D = 0.25 (margin +235 hp at 31,000 lb)
V_MAX = 204.0             # kt, zero of the linear law through acc_0 and G3 at 60 kt
# Figure 5.16, read: obstacle [ft] -> (optimum rotation speed [kt], minimum distance [ft])
FIG_5_16 = {50: (26.0, 325.0), 250: (30.0, 1150.0), 500: (38.0, 1850.0)}


@pytest.fixture(scope='module')
def ch4():
    path = pathlib.Path(__file__).parents[1] / 'performance' / 'test_chapter4.py'
    spec = importlib.util.spec_from_file_location('ch4_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _design(ch4):
    return {**ch4.REF_ROTOR, **ch4.EXAMPLE_DESIGN,
            **dict(load_elec=2200.0, flow_hyd=1.3, p_hyd=3000.0)}


def test_hover_power_at_28000_lb(ch4):
    p = ch4._hover_problem(GW=28000.0)
    assert_near_equal(p.get_val('P_req', units='hp')[0], P_HOVER_OGE, 1e-3)


def test_hermite_join_ends():
    p = _run(LowSpeedPowerComp(num_nodes=3), V=[0.0, 40 * KT, 40 * KT + 1e-6],
             V_b=40 * KT, P_hover=4200.0, P_b=2000.0, dP_b=-17.0)
    P = p.get_val('P_level')
    assert_near_equal(P[:2], [4200.0, 2000.0], 1e-12)
    assert_near_equal((P[2] - P[1]) / 1e-6, -17.0, 1e-4)


@pytest.fixture(scope='module')
def low_speed(ch4):
    V = np.array([0.0, 10.0, 20.0, 26.0, 30.0, 40.0])
    p = om.Problem()
    p.model.add_subsystem('g', LowSpeedPowerGroup(num_nodes=len(V)), promotes=['*'])
    p.setup()
    for k, v in _design(ch4).items():
        p.set_val(k, v)
    p.set_val('GW', 28000.0)
    p.set_val('CT_sigma', 0.12)
    p.set_val('alpha_F', np.deg2rad(-20.0) * np.ones(3))
    p.set_val('P_hover', P_HOVER_OGE)
    p.set_val('V', V, units='kn')
    p.run_model()
    return V, p


def test_low_speed_power_joins_the_chain(low_speed, ch4):
    """At 40 kt the join equals the forward flight chain with exact induced velocity."""
    V, p = low_speed
    q = om.Problem()
    from prouty.performance import ForwardFlightPowerGroup
    q.model.add_subsystem('ff', ForwardFlightPowerGroup(trim_options=dict(induced='exact')),
                          promotes=['*'])
    q.setup()
    for k, v in _design(ch4).items():
        q.set_val(k, v)
    q.set_val('GW', 28000.0)
    q.set_val('V', 40 * KT)
    q.set_val('alpha_F', np.deg2rad(-20.0))
    q.set_val('CT_sigma', 0.12)
    q.run_model()
    assert_near_equal(p.get_val('P_level')[-1], q.get_val('P_req', units='hp')[0], 1e-4)   # trim tolerance
    assert np.all(np.diff(p.get_val('P_level')) < 0)


@pytest.mark.parametrize('h', sorted(FIG_5_16))
def test_anchor_fig516_optimum_takeoff(ch4, h):
    """Optimum rotation speed within 8 kt of Figure 5.16; minimum distance 40-60 %
    of the print: the book's power at 25-40 kt and 28,000 lb is about 40 % higher
    (Figure 4.38 family, C4-29), which cuts its climb-out excess power."""
    p = om.Problem()
    p.model.add_subsystem('g', OptimumTakeoffGroup(), promotes=['*'])
    p.setup()
    for k, v in _design(ch4).items():
        p.set_val(k, v)
    for k, v in dict(GW=28000.0, CT_sigma=0.12, P_hover=P_HOVER_OGE, P_avail=P_AVAIL,
                     T_max_IGE=T_MAX_IGE, h=float(h)).items():
        p.set_val(k, v)
    p.set_val('alpha_F', np.deg2rad(-20.0) * np.ones(3))
    p.set_val('V_max', V_MAX, units='kn')
    p.run_model()
    v_book, x_book = FIG_5_16[h]
    assert abs(p.get_val('residual')) < 1e-6
    assert abs(p.get_val('V_rot', units='kn')[0] - v_book) < 8.0
    assert 0.4 < p.get_val('x_min')[0] / x_book < 0.6


@pytest.mark.parametrize('comp, inputs', [
    (LowSpeedPowerComp(num_nodes=3), {'V': [5.0, 30.0, 60.0]}),
    (JoinStencilComp(), {}),
    (JoinSlopeComp(), {'P_nodes': [2100.0, 2000.0, 1950.0]}),
    (TakeoffStencilComp(), {'x_nodes': [300.0, 290.0, 295.0]}),
])
def test_partials_join(comp, inputs):
    p = _run(comp, **inputs)
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-9, rtol=1e-9)
