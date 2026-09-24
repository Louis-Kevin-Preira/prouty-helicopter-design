"""Chapter 2, G5 -- VortexRingBoundariesComp, pp. 95, 102-107."""

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.vertical import FlowStatesGroup, VortexRingBoundariesComp

V_HOV_EX = np.sqrt(20000.0 / (2 * 0.002377 * 2827.4))    # 38.6 ft/s, example helicopter


def _run(V_D_bar, v_hov=V_HOV_EX, **opts):
    x = np.atleast_1d(np.asarray(V_D_bar, dtype=float))
    p = om.Problem()
    p.model.add_subsystem('g5', VortexRingBoundariesComp(num_nodes=x.size, **opts), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('V_D_bar', x)
    p.set_val('v_hov', np.full(x.size, v_hov), units='ft/s')
    p.run_model()
    return p


def test_example_boundaries_p102():
    """C2-2: 23 % and 125 % of v_1hov; book 400 and 2,900 ft/min."""
    p = _run(0.0)
    assert_near_equal(p.get_val('V_D_rough_high', units='ft/min')[0], 2900.0, 0.01)
    low = p.get_val('V_D_rough_low', units='ft/min')[0]
    assert_near_equal(low, 532.0, 0.005)
    assert low / 400.0 > 1.3


def test_classical_and_theoretical_boundaries():
    p = _run(0.0, v_hov=40.0)
    assert_near_equal(p.get_val('V_D_classic_high')[0], 80.0, 1e-14)
    assert_near_equal(p.get_val('V_D_max_instability')[0], 0.707 * 40.0, 1e-14)
    assert_near_equal(p.get_val('V_escape', units='kn')[0], 25.0, 1e-14)
    assert_near_equal(_run(0.0, V_escape_kt=10.0).get_val('V_escape', units='kn')[0], 10.0, 1e-14)


def test_margin_shape():
    lo, hi = 0.23, 1.25
    m = _run([lo, 0.5 * (lo + hi), hi, 0.0, 2.0]).get_val('vrs_margin')
    assert_near_equal(m[:3], [0.0, 1.0, 0.0], 1e-12)
    assert np.all(m[3:] < 0.0)


def test_rough_region_lies_on_the_chart_part_of_g0():
    """Above V_D_bar = 0.25 G0 is the Figure 2.13 data, i.e. the whole rough region but 0.23-0.25."""
    p = om.Problem()
    p.model.add_subsystem('g0', FlowStatesGroup(num_nodes=2), promotes=['*'])
    p.model.add_subsystem('g5', VortexRingBoundariesComp(num_nodes=2), promotes=['*'])
    p.setup()
    p.set_val('T', [20000.0, 20000.0])
    p.set_val('A', 2827.4)
    p.set_val('V_c', [-0.5 * V_HOV_EX, -1.0 * V_HOV_EX])
    p.run_model()
    assert np.all(p.get_val('vrs_margin') > 0.0)
    assert np.all(p.get_val('v1_bar') > p.get_val('V_D_bar') / 2 + np.sqrt(p.get_val('V_D_bar') ** 2 / 4 + 1))


def test_partials():
    p = _run([0.1, 0.7, 1.5], v_hov=39.0)
    data = p.check_partials(method='cs', compact_print=True, out_stream=None)
    assert_check_partials(data, atol=1e-12, rtol=1e-12)
