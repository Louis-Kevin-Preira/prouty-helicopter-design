"""Chapter 5, G2d -- height-velocity diagram, pp. 352-358."""
import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.special_performance import (LowHoverHeightComp, MinPowerSpeedComp,
                                        CriticalSpeedComp, MultiEngineCriticalSpeedComp,
                                        HighHoverHeightComp, HVBoundaryComp,
                                        HeightVelocityGroup)
from prouty.special_performance.critical_speed_comp import FIG_5_9_TOP, KT
from prouty.special_performance.hv_boundary_comp import S_UP, X_UP, S_LO, X_LO

RHO_0 = 0.002377
EXAMPLE = dict(GW=20000.0, f=20.0, A=np.pi * 30.0 ** 2, A_b=240.0, V_tip=650.0,
               C_d=0.01, e=0.8)


def _run(system, **inputs):
    p = om.Problem()
    p.model.add_subsystem('s', system, promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in inputs.items():
        p.set_val(k, v[0], units=v[1]) if isinstance(v, tuple) else p.set_val(k, v)
    p.run_model()
    return p


def _power(V, GW, f, A, A_b, V_tip, C_d, e, rho_ratio=1.0):
    """Main rotor power of p. 356, hp/(rho/rho_0)."""
    return ((GW / rho_ratio) ** 2 / (1100 * e * RHO_0 * A * V) + RHO_0 * f * V ** 3 / 1100
            + RHO_0 * A_b * V_tip ** 3 * C_d / 8 / 550 * (1 + 3 * (V / V_tip) ** 2))


# ---------- speed for minimum power ----------

def test_min_power_speed_minimizes_the_power_of_p356():
    V = _run(MinPowerSpeedComp(), **EXAMPLE).get_val('V_min')[0]
    h = 1e-3
    dP = (_power(V + h, **EXAMPLE) - _power(V - h, **EXAMPLE)) / (2 * h)
    assert abs(dP) < 1e-6
    assert _power(V, **EXAMPLE) < min(_power(0.9 * V, **EXAMPLE), _power(1.1 * V, **EXAMPLE))


def test_min_power_speed_example_near_83_kt():
    """Fig. 5.10 noses (80 kt FAA, 101 kt military) both point to V_min of about
    82-84 kt through Fig. 5.9; the energy method gives that for e = 0.8."""
    V = _run(MinPowerSpeedComp(), **EXAMPLE).get_val('V_min', units='kn')
    assert 80.0 < V < 87.0


# ---------- Figure 5.9 ----------

@pytest.mark.parametrize('td', ['faa', 'military'])
def test_fig59_top_lines_reproduced(td):
    """At C_L/sigma = 0..20 the fitted lines pass through the digitized points."""
    a, b = FIG_5_9_TOP[td]
    cl = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    ct = 0.083
    mu = np.sqrt(2 * ct / cl[1:])
    V_min_kt = a[1:] + b[1:] * 100.0            # V_CR = 100 kt on each line
    p = _run(CriticalSpeedComp(num_nodes=4, time_delay=td), CT_sigma=ct * np.ones(4),
             V_min=mu * 650.0, V_tip=650.0 * np.ones(4))
    V_CR = (mu * 650.0 / KT - a[1:]) / b[1:]
    assert_near_equal(p.get_val('CL_sigma'), cl[1:], 1e-12)
    assert_near_equal(p.get_val('V_CR'), V_CR, 1e-10)


@pytest.mark.parametrize('td, v_cr, h_hi', [('faa', 80.0, 1330.0), ('military', 101.0, 1390.0)])
def test_anchor_fig510_high_hover_height(td, v_cr, h_hi):
    """Figure 5.10, sea level: noses at 80 / 101 kt, tops at 1,330 / 1,390 ft."""
    p = _run(HighHoverHeightComp(time_delay=td), V_CR=v_cr)
    assert_near_equal(p.get_val('h_hi'), h_hi, 0.04)


@pytest.mark.parametrize('td, v_cr', [('faa', 80.0), ('military', 101.0)])
def test_anchor_fig510_critical_speed_from_example(td, v_cr):
    """Chain V_min (energy method) -> V_CR against the noses of Figure 5.10.

    V_min = 84.3 kt here; the noses imply 82-83 kt, and dV_CR/dV_min = 2.8,
    so 2 % on V_min becomes 5-9 % on V_CR (e, f, C_d of the example are readings)."""
    p = om.Problem()
    p.model.add_subsystem('g', HeightVelocityGroup(time_delay=td), promotes=['*'])
    p.setup()
    for k, v in EXAMPLE.items():
        p.set_val(k, v)
    p.set_val('CW_sigma', 0.083)
    p.run_model()
    assert_near_equal(p.get_val('V_CR', units='kn'), v_cr, 0.10)


# ---------- low hover height ----------

def test_low_hover_height_single_equals_multi_without_power():
    args = dict(V_LG=8.0, CW_sigma=0.083, P_IGE=1500.0)
    single = _run(LowHoverHeightComp(), **args).get_val('h_lo')
    multi = _run(LowHoverHeightComp(engines='multi'), P_avail=0.0, **args).get_val('h_lo')
    assert_near_equal(multi, single, 1e-12)
    assert_near_equal(single, 8.0 * 11735.0 * 21.67 ** 2 * (1 - 0.083 / 0.2) / (1100 * 1500.0),
                      1e-12)


def test_c5_3_book_option():
    args = dict(V_LG=8.0, CW_sigma=0.083, P_IGE=2500.0, P_avail=2000.0)
    book = _run(LowHoverHeightComp(engines='multi', book=True), **args).get_val('h_lo')
    coh = _run(LowHoverHeightComp(engines='multi'), **args).get_val('h_lo')
    assert_near_equal(book / coh, (1 - np.sqrt(0.083 / 0.2)) / (1 - 0.083 / 0.2), 1e-12)


# ---------- multiengine ----------

@pytest.mark.parametrize('td, k', [('faa', 0.5), ('military', 1.0)])
def test_multi_engine_critical_speed_and_height(td, k):
    p = _run(MultiEngineCriticalSpeedComp(num_nodes=2, time_delay=td), V_sink=40.0,
             P_req=[2300.0, 2100.0], P_avail=2000.0, GW=20000.0, h_lo=120.0)
    assert_near_equal(p.get_val('V_CR'), 40.0 * k, 1e-12)
    assert_near_equal(p.get_val('RD'), [550 * 300 / 20000, 550 * 100 / 20000], 1e-12)
    assert_near_equal(p.get_val('h_CR'), 120.0, 1e-4)
    p = _run(MultiEngineCriticalSpeedComp(), h_lo=10.0)
    assert_near_equal(p.get_val('h_CR'), 50.0, 1e-3)


# ---------- boundary ----------

def test_boundary_ends_and_figure_58_points():
    p = _run(HVBoundaryComp(num_nodes=2), s=[0.0, 1.0], h_lo=12.0, h_hi=1330.0, h_CR=95.0,
             V_CR=80.0)
    assert_near_equal(p.get_val('h_up'), [1330.0, 95.0], 1e-12)
    assert_near_equal(p.get_val('h_lo_branch'), [12.0, 95.0], 1e-12)
    assert_near_equal(p.get_val('V_up'), [0.0, 80.0], 1e-12)
    assert_near_equal(p.get_val('V_lo'), [0.0, 80.0], 1e-12)
    n = len(S_UP)
    p = _run(HVBoundaryComp(num_nodes=n), s=S_UP, V_CR=1.0)
    assert_near_equal(p.get_val('V_up'), X_UP, 1e-10)
    s = np.linspace(0, 1, 201)
    p = _run(HVBoundaryComp(num_nodes=201), s=s, V_CR=1.0)
    assert np.all(np.diff(p.get_val('V_up')) >= -1e-9)
    assert np.all(np.diff(p.get_val('V_lo')) >= -1e-9)


# ---------- derivatives ----------

@pytest.mark.parametrize('system, inputs', [
    (LowHoverHeightComp(), {}),
    (LowHoverHeightComp(engines='multi', book=True), {'P_IGE': 2500.0, 'P_avail': 2000.0}),
    (MinPowerSpeedComp(num_nodes=2), {'GW': [20000.0, 16000.0], 'rho_ratio': [1.0, 0.8]}),
    (CriticalSpeedComp(num_nodes=2, time_delay='military'), {'V_min': [140.0, 110.0]}),
    (MultiEngineCriticalSpeedComp(num_nodes=2), {'h_lo': 49.5}),
    (HighHoverHeightComp(num_nodes=2), {'V_CR': [35.0, 95.0]}),
    (HVBoundaryComp(num_nodes=7), {}),
    (HeightVelocityGroup(num_points=5), {'CW_sigma': 0.083}),
    (HeightVelocityGroup(engines='multi', time_delay='military', num_points=5), {'h_lo': 60.0}),
])
def test_partials(system, inputs):
    p = _run(system, **inputs)
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-8, rtol=1e-8)


# ---------- Figure 5.10, 4,000 ft / 95 F panels (block A3) ----------

from prouty.special_performance.book_figures import FIG_5_10_HOT, RHO_RATIO_4000_95F


def _hv_hot(td):
    p = om.Problem(reports=False)
    p.model.add_subsystem('g', HeightVelocityGroup(time_delay=td), promotes=['*'])
    p.setup()
    for k, v in EXAMPLE.items():
        p.set_val(k, v)
    p.set_val('rho_ratio', RHO_RATIO_4000_95F)
    p.set_val('CW_sigma', 0.083 / RHO_RATIO_4000_95F)
    p.run_model()
    return p


@pytest.mark.parametrize('td, tol_v, tol_h', [('faa', 0.10, 0.18), ('military', 0.03, 0.07)])
def test_anchor_fig510_hot_dual_engine(td, tol_v, tol_h):
    """4,000 ft, 95 F, dual engine failure: V_CR 115 kt (FAA, printed 107) and
    131.7 kt (military, printed 132); h_hi 2,622 / 2,414 ft (printed 2,260 / 2,300).
    The FAA excess follows V_CR, as at sea level (87 vs 80 kt): read at the printed
    noses, Figure 5.9 gives 2,316 / 2,423 ft, within 2.5 / 5 %."""
    v_cr, _, h_hi = FIG_5_10_HOT[td]['dual']
    p = _hv_hot(td)
    assert_near_equal(p.get_val('V_CR', units='kn'), v_cr, tol_v)
    assert_near_equal(p.get_val('h_hi'), h_hi, tol_h)
    q = _run(HighHoverHeightComp(time_delay=td), V_CR=v_cr)
    assert_near_equal(q.get_val('h_hi'), h_hi, 0.06)


@pytest.mark.parametrize('td', ['faa', 'military'])
def test_anchor_fig510_hot_single_engine(td):
    """4,000 ft, 95 F, one engine out (twin): hover OGE 2,576 hp and one-engine takeoff
    rating 1,598 hp (Chapter 4 hover, isothermal day), sink 6 ft/s. V_sink = 26 kt:
    V_CR = 13 kt (FAA, printed 16) and 26 kt (military, printed 34); h_hi read on
    Figure 5.9 at V_CR (assumption, no method in the book) = 293 / 300 ft against
    296 / 322 printed."""
    from prouty.special_performance import MultiEngineSinkGroup
    from prouty.special_performance.book_figures import EXAMPLE_ROTOR, EXAMPLE_POWER
    p = om.Problem(reports=False)
    p.model.add_subsystem('g', MultiEngineSinkGroup(time_delay=td), promotes=['*'])
    p.setup()
    for k, v in {**{k: v for k, v in EXAMPLE_ROTOR.items() if k != 'GW'}, **EXAMPLE_POWER}.items():
        p.set_val('power.' + k, v)
    p.set_val('power.rho', 0.002377 * RHO_RATIO_4000_95F)
    p.set_val('power.V_son', 1154.6)
    p.set_val('power.CT_sigma', 0.105)
    p.set_val('power.alpha_F', np.deg2rad(-15.0) * np.ones(3))
    for k, v in dict(P_hover=2576.4, P_avail=1598.0, V_LG=6.0, h_lo=30.0).items():
        p.set_val(k, v)
    p.run_model()
    v_cr, _, h_hi = FIG_5_10_HOT[td]['single']
    V_CR = p.get_val('V_CR', units='kn')[0]
    assert 0.6 * v_cr < V_CR < v_cr
    q = _run(HighHoverHeightComp(time_delay=td), V_CR=V_CR)
    assert_near_equal(q.get_val('h_hi'), h_hi, 0.08)


def test_hot_day_hover_inputs_from_chapter4():
    """The two powers above: Chapter 4 hover at 4,000 ft on the isothermal 95 F day."""
    import importlib.util, pathlib
    path = pathlib.Path(__file__).parents[1] / 'performance' / 'test_chapter4.py'
    spec = importlib.util.spec_from_file_location('ch4_tests', path)
    ch4 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ch4)
    p = ch4._hover_problem(day='isothermal', altitude=4000.0)
    assert_near_equal(p.get_val('P_req', units='hp')[0], 2576.4, 1e-3)
    assert_near_equal(p.get_val('P_avail', units='hp')[0][0] / 2.0, 1598.0, 1e-3)
    assert_near_equal(p.get_val('rho')[0] / 0.002377, RHO_RATIO_4000_95F, 1e-3)
