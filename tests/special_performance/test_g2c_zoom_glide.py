"""Chapter 5, G2c -- zoom maneuver and glide distance, pp. 351-352."""
import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.special_performance import (ZoomClimbAngleComp, ZoomAltitudeGainComp,
                                        GlideDistanceComp, ZoomGlideGroup)

KT = 1.6878

# Figure 5.6 (p. 353), power failure at 160 kt, digitized: V_1 [kt] -> Delta_h, Delta_d [ft]
FIG_5_6 = {70: (453, 2007), 80: (430, 2178), 90: (397, 2236), 100: (353, 2018),
           120: (249, 1457), 140: (125, 785)}
# Figure 5.5 (p. 351), rate of descent in autorotation, digitized: V [kt] -> R/D [ft/min]
FIG_5_5 = {70: 1626, 80: 1605, 90: 1637, 100: 1711, 120: 2047, 140: 2358}


def _run(system, **inputs):
    p = om.Problem()
    p.model.add_subsystem('s', system, promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in inputs.items():
        p.set_val(k, v[0], units=v[1]) if isinstance(v, tuple) else p.set_val(k, v)
    p.run_model()
    return p


def test_climb_angle_example():
    """C_W/sigma = 0.083 with the conservative (C_T/sigma)_max = 0.12 (p. 352)."""
    g = _run(ZoomClimbAngleComp()).get_val('gamma_c', units='deg')
    assert_near_equal(g, np.degrees(np.arccos(0.083 / 0.12)), 1e-12)


@pytest.mark.parametrize('v1', sorted(FIG_5_6))
def test_anchor_fig56_glide_ratio_matches_fig55(v1):
    """Figures 5.5 and 5.6 together: Delta_d = Delta_h V_1/(R/D) within 5 %."""
    dh, dd = FIG_5_6[v1]
    p = _run(GlideDistanceComp(), V_1=(v1, 'kn'), RD=(FIG_5_5[v1], 'ft/min'), delta_h=dh)
    assert_near_equal(p.get_val('delta_d'), dd, 0.05)


def test_altitude_gain_limits():
    """No gain at V_1 = V_0; pure kinetic energy exchange with no power."""
    p = _run(ZoomAltitudeGainComp(num_nodes=2), V_0=270.0, V_1=[270.0, 150.0],
             P_0=0.0, P_1=[0.0, 0.0])
    assert_near_equal(p.get_val('delta_h'), [0.0, (270.0 ** 2 - 150.0 ** 2) / 64.4], 1e-12)
    p = _run(ZoomAltitudeGainComp(), V_0=270.0, V_1=270.0, P_0=3000.0, P_1=3000.0)
    assert_near_equal(p.get_val('delta_h'), 0.0, 1e-12)


def test_fig56_implies_plausible_power():
    """Back-solving Figure 5.6 gives hp_0 + hp_1 of 5,000 to 7,500 hp, rising with V_1,
    i.e. about 3,900 hp at 160 kt: the 20,000 lb curve of Figure 4.38 (p. 319)."""
    gc = np.arccos(0.083 / 0.12)
    k = 550.0 / (2 * 20000.0 * 32.2 * np.sin(gc))
    V0 = 160 * KT
    sums = []
    for v1 in (70, 90, 120, 140):
        dh = FIG_5_6[v1][0]
        V1 = v1 * KT
        sums.append(((V0 ** 2 - V1 ** 2) / 64.4 - dh) / (k * (V0 - V1)))
    assert np.all(np.diff(sums) > 0)
    assert 4500 < sums[0] < 6000 and 6500 < sums[-1] < 8000


def test_group_and_partials():
    p = om.Problem()
    p.model.add_subsystem('g', ZoomGlideGroup(num_nodes=3), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.set_val('V_0', 160, units='kn')
    p.set_val('P_0', 3900.0)
    p.set_val('V_1', [70, 90, 120], units='kn')
    p.set_val('P_1', [1150.0, 1500.0, 2300.0])
    p.set_val('RD', [1626, 1637, 2047], units='ft/min')
    p.set_val('h_0', 500.0)
    p.run_model()
    assert np.all(p.get_val('delta_h') > 0)
    assert np.all(p.get_val('glide_distance') > p.get_val('delta_d'))
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-9, rtol=1e-9)
