"""Chapter 5, G6 -- return-to-target maneuver, pp. 368-371."""
import importlib.util
import pathlib

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.special_performance.book_figures import FIG_5_17
from prouty.special_performance import (TurnDecelerationComp, TurnDecelerationGroup,
                                        AutorotationLimitComp, HoverPowerScalingComp,
                                        ReturnToTargetComp, ReturnToTargetChainGroup)

KT = 1.6878
V_GRID = np.array([26, 30, 35, 40, 50, 60, 70, 80, 90, 100, 115, 125.0]) * KT
M = len(V_GRID)


@pytest.fixture(scope='module')
def ch4():
    path = pathlib.Path(__file__).parents[1] / 'performance' / 'test_chapter4.py'
    spec = importlib.util.spec_from_file_location('ch4_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _chain(ch4, powered_turn='exact', num_steps=40, n_p=1.15):
    p = om.Problem()
    p.model.add_subsystem('g', ReturnToTargetChainGroup(V_grid=V_GRID, num_steps=num_steps,
                                                        powered_turn=powered_turn),
                          promotes=['*'])
    p.setup()
    p.set_val('sigma', 0.0849)
    p.set_val('cd_bar', 0.01 * np.ones(M))
    p.set_val('R', 30.0)
    p.set_val('V_0', 115 * KT)
    if powered_turn == 'exact':
        for k, v in ch4.EXAMPLE_HOVER.items():
            if k not in ('altitude', 'GW'):
                p.set_val('powered.hover.' + k, v)
        p.set_val('powered.hover.altitude', 0.0)
        p.set_val('powered.hover.theta_0', 17.0)
        p.set_val('powered.hover.rotor_ref.theta_0', 13.0)
        p.set_val('powered.hover.tr_theta_0', 15.0)
        for k, v in {**ch4.REF_ROTOR, **ch4.EXAMPLE_DESIGN,
                     **dict(load_elec=2200.0, flow_hyd=1.3, p_hyd=3000.0)}.items():
            if k != 'GW':
                p.set_val('powered.power.' + k, v)
        p.set_val('powered.power.CT_sigma', 0.12)
        p.set_val('powered.power.alpha_F', np.deg2rad(-20.0) * np.ones(3))
    else:
        p.set_val('n_p', n_p)
    p.run_model()
    return p


@pytest.fixture(scope='module')
def exact(ch4):
    return _chain(ch4)


# ---------------- turn table ----------------

def test_turn_table_zero_torque_at_the_ceiling(exact):
    p = exact
    assert np.max(np.abs(p.get_val('decel.CQ_sigma'))) < 1e-10
    n = p.get_val('decel.n_turn')
    assert_near_equal(n[-2], 2.0, 0.03)                       # 115 kt, transient band
    assert n[0] < 1.0 < n[1]                                  # autorotative limit near 28 kt


def test_autorotative_limit_and_powered_turn(exact):
    """n_turn = 1 at 28.4 kt; the powered turn there holds n_p = 1.89 at 4,077 hp."""
    p = exact
    assert_near_equal(p.get_val('limit.V_sw', units='kn'), 28.4, 0.02)
    assert_near_equal(p.get_val('powered.power.P_level'), p.get_val('powered.hover.P_rating'),
                      1e-8)
    assert 1.5 < p.get_val('powered.n_p')[0] < 2.2


def test_anchor_fig517(exact):
    """Minimum speed as on Figure 5.17 (26.6 kt against 27 kt). The maneuver is
    faster and tighter: turn 9.8 s against ~12.5 s, straight return 6.7 s against
    ~10.5 s (G3 acceleration above Fig. 5.14), total 16.5 s against 23 s, loop
    x 560 ft, y 770 ft against ~1,010 and ~965 ft: the transient thrust ceiling
    of Fig. 5.2 decelerates harder than the book's zero-torque upper stall limit."""
    p = exact
    assert_near_equal(p.get_val('V_min', units='kn'), FIG_5_17['V_min_kt'], 0.05)
    assert 0.7 < p.get_val('t1')[0] / FIG_5_17['t_turn'] < 0.85
    assert 0.65 < p.get_val('t_total')[0] / FIG_5_17['t_total'] < 0.8
    assert 0.5 < p.get_val('x').max() / FIG_5_17['x_max'] < 0.65
    assert 0.75 < p.get_val('y').max() / FIG_5_17['y_max'] < 0.85


def test_final_heading_and_arrival(exact):
    p = exact
    assert abs(p.get_val('maneuver.r_point')) < 1e-8
    assert abs(p.get_val('maneuver.r_dist')) < 1e-6
    x1, y1 = p.get_val('x')[-1], p.get_val('y')[-1]
    assert x1 > 0 and y1 > 0                                  # target seen in the 3rd quadrant
    psi1 = np.pi + np.arctan(y1 / x1)
    assert np.pi < psi1 < 1.5 * np.pi


def test_fixed_powered_turn_fallback(ch4):
    p = _chain(ch4, powered_turn='fixed', n_p=1.15)
    assert abs(p.get_val('maneuver.r_point')) < 1e-8
    assert p.get_val('t1')[0] > 10.5                           # weaker turn, longer phase 1


def test_step_convergence(ch4):
    """N = 40 within 0.5 % of N = 80 on the total time (16.53 / 16.53 s exact,
    17.36 / 17.43 s fixed)."""
    t40 = _chain(ch4, 'exact', 40).get_val('t_total')[0]
    t80 = _chain(ch4, 'exact', 80).get_val('t_total')[0]
    assert abs(t40 / t80 - 1) < 0.005


# ---------------- derivatives ----------------

def test_partials_components():
    for comp, inputs in [
        (TurnDecelerationComp(num_nodes=2), {'alpha_TPP': [0.2, 0.6], 'CH_sigma': [0.001, 0.0]}),
        (HoverPowerScalingComp(), {'n': 1.7}),
    ]:
        p = om.Problem()
        p.model.add_subsystem('c', comp, promotes=['*'])
        p.setup(force_alloc_complex=True)
        for k, v in inputs.items():
            p.set_val(k, v)
        p.run_model()
        assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                              atol=1e-9, rtol=1e-9)


def test_analytic_partials_against_fd():
    """Analytic partials (tangent through the Heun steps; implicit function theorem for
    V_sw) against central finite differences. Akima tables are not complex-step safe."""
    n_tab = np.array([0.78, 1.13, 1.43, 1.62, 1.83, 1.92, 1.97, 1.99, 2.0, 2.01, 2.0, 1.99])
    vdot = np.array([-59.7, -53.7, -46.1, -39.4, -29.0, -21.9, -17.2, -13.9, -11.7, -10.2,
                     -8.6, -8.0])
    acc = np.linspace(30.0, 8.0, M)
    for comp, inputs in [
        (AutorotationLimitComp(V_grid=V_GRID), {'n_tab': n_tab}),
        (ReturnToTargetComp(V_grid=V_GRID, num_steps=20),
         {'n_tab': n_tab, 'Vdot_tab': vdot, 'acc_tab': acc, 'V_sw': 48.0, 'n_p': 1.8}),
    ]:
        p = om.Problem()
        p.model.add_subsystem('c', comp, promotes=['*'])
        p.setup(force_alloc_complex=True)
        for k, v in inputs.items():
            p.set_val(k, v)
        p.run_model()
        data = p.check_partials(method='fd', form='central', step=1e-6, compact_print=True,
                                out_stream=None)
        assert_check_partials(data, atol=1e-5, rtol=1e-5)
