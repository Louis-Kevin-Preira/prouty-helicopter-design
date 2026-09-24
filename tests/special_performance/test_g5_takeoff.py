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
