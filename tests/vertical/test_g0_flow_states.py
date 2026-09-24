"""Chapter 2, G0 -- FlowStatesGroup, pp. 93-95 and Figure 2.13 p. 113."""

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import (assert_check_partials, assert_check_totals,
                                         assert_near_equal)

from prouty.vertical import FIGURE_2_13, AxialInducedVelocityComp, FlowStatesGroup

# Example helicopter, Appendix A: 20,000 lb, A = 2,827 ft^2, theta_1 = -10 deg; sea level
GW, A_M, RHO_SL = 20000.0, 2827.0, 0.002377


def _run_group(V_c, theta_1_deg=-10.0, T=GW, **opts):
    V_c = np.atleast_1d(np.asarray(V_c, dtype=float))
    nn = V_c.size
    p = om.Problem()
    p.model.add_subsystem('g0', FlowStatesGroup(num_nodes=nn, **opts), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('T', np.full(nn, T), units='lbf')
    p.set_val('rho', RHO_SL, units='slug/ft**3')
    p.set_val('A', A_M, units='ft**2')
    p.set_val('V_c', V_c, units='ft/s')
    p.set_val('theta_1', theta_1_deg, units='deg')
    p.run_model()
    return p


def _v1_bar(x, theta_1_deg=-10.0, **opts):
    x = np.atleast_1d(np.asarray(x, dtype=float))
    p = om.Problem()
    p.model.add_subsystem('c', AxialInducedVelocityComp(num_nodes=x.size, **opts),
                          promotes=['*'])
    p.setup()
    p.set_val('V_D_bar', x)
    p.set_val('theta_1', theta_1_deg, units='deg')
    p.run_model()
    return p.get_val('v1_bar')


# ------------------------------------------------------------------------
# book anchors
# ------------------------------------------------------------------------

def test_hover_induced_velocity_example():
    """p. 93 equation; p. 101 quotes 39 ft/s for the example helicopter."""
    v_hov = _run_group(0.0).get_val('v_hov')[0]
    assert_near_equal(v_hov, 39.0, 0.02)


def test_climb_is_momentum_p94():
    """v_1c = -V_c/2 + sqrt((V_c/2)^2 + v_1hov^2); 8.33 ft/s is the 500 ft/min of p. 100."""
    V_c = np.array([8.33, 20.0, 50.0])
    p = _run_group(V_c)
    v_hov = p.get_val('v_hov')
    assert_near_equal(p.get_val('v1'), -V_c / 2 + np.sqrt((V_c / 2) ** 2 + v_hov ** 2), 1e-12)


def test_hover_limit():
    assert_near_equal(_run_group(0.0).get_val('v1_bar'), [1.0], 1e-14)


def test_windmill_brake_is_momentum_p95():
    """v_1WB = V_D/2 - sqrt((V_D/2)^2 - v_1hov^2) beyond the high window."""
    x = np.array([3.6, 4.0, 6.0])
    assert_near_equal(_v1_bar(x), x / 2 - np.sqrt(x ** 2 / 4 - 1), 1e-12)


def test_figure_2_13_autorotation_reading_p112():
    """theta_1 = -10 deg, V_D_bar - v1_bar = 0.22 --> V_D_bar = 1.97."""
    lo, hi = 1.9, 2.05
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mid - _v1_bar(mid)[0] < 0.22 else (lo, mid)
    assert_near_equal(0.5 * (lo + hi), 1.97, 0.01)


@pytest.mark.parametrize('theta_1_deg', [0.0, -12.0])
def test_chart_nodes_reproduced(theta_1_deg):
    """Between the windows the model passes through the digitized points."""
    x, y = FIGURE_2_13[theta_1_deg]
    keep = (x >= 0.25) & (x <= 2.6)
    assert_near_equal(x[keep] - _v1_bar(x[keep], theta_1_deg), y[keep], 1e-10)


# ------------------------------------------------------------------------
# internal consistency
# ------------------------------------------------------------------------

@pytest.mark.parametrize('edge', [0.0, 0.25, 2.6, 3.6])
def test_blend_is_c1_at_window_edges(edge):
    eps = 1e-6
    v = _v1_bar([edge - 2 * eps, edge - eps, edge + eps, edge + 2 * eps])
    assert abs(v[2] - v[1]) < 1e-4
    slope_l, slope_r = (v[1] - v[0]) / eps, (v[3] - v[2]) / eps
    assert abs(slope_r - slope_l) < 1e-3 * max(1.0, abs(slope_l))


def test_twist_interpolation_is_linear():
    x = np.array([0.8, 1.6, 2.2])
    assert_near_equal(_v1_bar(x, -6.0), 0.5 * (_v1_bar(x, 0.0) + _v1_bar(x, -12.0)), 1e-12)


def test_invalid_windows_raise():
    p = om.Problem()
    p.model.add_subsystem('c', AxialInducedVelocityComp(high_window=(1.5, 2.5)))
    with pytest.raises(ValueError):
        p.setup()


def test_promoted_defaults_are_consistent_with_chapter4():
    """G0 shares T, rho, A, V_c, v_hov with the Chapter 4 vertical climb."""
    from prouty.performance import VerticalClimbPowerComp
    p = om.Problem()
    p.model.add_subsystem('g0', FlowStatesGroup(num_nodes=2), promotes=['*'])
    p.model.add_subsystem('power', VerticalClimbPowerComp(num_nodes=2),
                          promotes_inputs=['rho', 'V_c', 'v_hov'])
    p.setup()
    p.final_setup()


# ------------------------------------------------------------------------
# derivatives
# ------------------------------------------------------------------------

@pytest.mark.parametrize('theta_1_deg, opts', [
    (-10.0, {}),
    (-4.0, {'low_window': (-0.1, 0.4), 'high_window': (2.4, 3.2)})])
def test_partials(theta_1_deg, opts):
    x = np.array([-2.0, 0.0, 0.12, 0.3, 1.0, 1.38, 1.7, 1.81, 1.97, 2.5, 2.9, 3.3, 4.5])
    p = _run_group(-x * 38.6, theta_1_deg, **opts)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-8, rtol=1e-8)


def test_totals():
    p = _run_group([-60.0, 10.0])
    data = p.check_totals(of=['v1'], wrt=['T', 'V_c', 'theta_1'], method='cs', out_stream=None)
    assert_check_totals(data, atol=1e-8, rtol=1e-8)
