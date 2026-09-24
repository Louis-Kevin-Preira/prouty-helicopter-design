"""Chapter 5, G7 -- towing, pp. 371-372."""
import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.special_performance import TowingGroup, TowlineTensionComp


def _run(nn=1, **inputs):
    p = om.Problem()
    p.model.add_subsystem('g', TowingGroup(num_nodes=nn), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in inputs.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_anchor_p372_example_tension():
    """27,800 lb max thrust OGE, 17,000 lb, flat towline: 22,000 lb (p. 372)."""
    p = _run(T_max=27800.0, GW=17000.0, gamma=0.0)
    assert_near_equal(p.get_val('tension', units='lbf'), 22000.0, 0.005)


@pytest.mark.parametrize('tw, gamma_deg, ratio', [
    (1.2, 0.0, 0.66), (1.2, 45.0, 0.26),      # bottom curve of Figure 5.18
    (2.4, 0.0, 2.18), (2.4, 45.0, 1.60),      # top curve
])
def test_anchor_fig518(tw, gamma_deg, ratio):
    """Curve ends read on Figure 5.18 (p. 372)."""
    p = _run(T_max=tw * 10000.0, GW=10000.0, gamma=np.deg2rad(gamma_deg))
    assert_near_equal(p.get_val('tension_ratio'), ratio, 0.03)


def test_force_balance():
    """Rotor thrust needed to hold the computed tension equals T_max."""
    g = np.deg2rad([0.0, 15.0, 40.0])
    p = _run(3, T_max=24000.0, GW=18000.0, gamma=g)
    F = p.get_val('tension')
    T = np.hypot(18000.0 + F * np.sin(g), F * np.cos(g))
    assert_near_equal(T, 24000.0 * np.ones(3), 1e-12)


def test_partials():
    p = _run(3, T_max=24000.0, GW=18000.0, gamma=np.deg2rad([0.0, 15.0, 40.0]))
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-9, rtol=1e-9)
