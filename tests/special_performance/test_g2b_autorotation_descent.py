"""Chapter 5, G2b -- steady descent in autorotation (pp. 350-351), and G2c
connected to G2b and Chapter 4 (p. 352)."""
import importlib.util
import pathlib

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.special_performance.book_figures import EXAMPLE_ROTOR as REF, FIG_5_5, FIG_5_6
FIG_5_6_DH = {v: dh for v, (dh, _) in FIG_5_6.items() if v in (60, 80, 100, 120, 140)}
from prouty.special_performance import (AutorotationDescentGroup, BestAutorotationSpeedGroup,
                                        DescentSpeedStencilComp, SpeedNodesComp,
                                        ChainSplitComp, ZoomGlideChainGroup)


def _set(p, values):
    for k, v in values.items():
        p.set_val(k, v)


@pytest.fixture(scope='module')
def sweep():
    V = np.array(sorted(FIG_5_5))
    p = om.Problem()
    p.model.add_subsystem('g', AutorotationDescentGroup(num_nodes=len(V)), promotes=['*'])
    p.setup()
    _set(p, REF)
    p.set_val('V', V, units='kn')
    p.run_model()
    return V, p


def test_trim_residuals_vanish(sweep):
    _, p = sweep
    for name in ('res_alpha_F', 'res_CT_sigma', 'res_CQ_sigma'):
        assert np.max(np.abs(p.get_val(name))) < 1e-10


def test_anchor_fig55_rate_of_descent(sweep):
    """Within 10 % of Figure 5.5 from 70 to 140 kt (chain below the figure above 100 kt)."""
    V, p = sweep
    rd = p.get_val('RD', units='ft/min')
    fig = np.array([FIG_5_5[v] for v in V])
    assert np.all(np.abs(rd / fig - 1) < 0.10)


@pytest.mark.parametrize('target, v_kt, tol', [('min_rate', 82.0, 0.08), ('min_angle', 125.0, 0.06)])
def test_anchor_fig55_best_speeds(target, v_kt, tol):
    """Bottom of the curve about 80-85 kt; tangent ray at about 125 kt (Figure 5.5)."""
    p = om.Problem()
    p.model.add_subsystem('g', BestAutorotationSpeedGroup(target=target), promotes=['*'])
    p.setup()
    _set(p, REF)
    p.run_model()
    assert abs(p.get_val('residual')) < 1e-8
    assert_near_equal(p.get_val('V', units='kn'), v_kt, tol)
    if target == 'min_angle':
        # p. 351: maximum L/D of the example = 6.3
        assert_near_equal(p.get_val('LD_nodes')[1], 6.3, 0.06)
        ld = p.get_val('LD_nodes')
        assert ld[1] >= max(ld[0], ld[2]) - 1e-6


# ---------------- G2c chain ----------------

@pytest.fixture(scope='module')
def ch4():
    path = pathlib.Path(__file__).parents[1] / 'performance' / 'test_chapter4.py'
    spec = importlib.util.spec_from_file_location('ch4_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def chain(ch4):
    V1 = np.array([60.0, 70.0, 80.0, 87.0, 90.0, 100.0, 120.0, 140.0])
    p = om.Problem()
    p.model.add_subsystem('g', ZoomGlideChainGroup(num_nodes=len(V1)), promotes=['*'])
    p.setup()
    _set(p, {**ch4.REF_ROTOR, **ch4.EXAMPLE_DESIGN,
             **dict(load_elec=2200.0, flow_hyd=1.3, p_hyd=3000.0)})
    p.set_val('V_1', V1, units='kn')
    p.set_val('V_0', 160.0, units='kn')
    p.set_val('CW_sigma', 0.083)
    p.run_model()
    return V1, p


def test_chain_powers_are_the_chapter4_powers(chain, ch4):
    V1, p = chain
    ref = ch4._ff_power(100.0).get_val('P_req', units='hp')[0]
    assert_near_equal(p.get_val('P_1', units='hp')[V1 == 100.0][0], ref, 1e-4)


def test_chain_best_autorotation_speed_for_glide(chain):
    """Figure 5.6: extra glide distance peaks near 87 kt."""
    V1, p = chain
    dd = p.get_val('delta_d')
    assert 80.0 <= V1[np.argmax(dd)] <= 95.0


def test_chain_altitude_gain_above_print(chain):
    """Chapter 4 powers are ~35 % below Figure 4.38 (C4-29), and 1,876 hp at 160 kt
    against about 3,900 hp behind Figure 5.6 (C5-7): less power is dissipated
    in the zoom, so Delta_h is 40-60 % above the print, never below."""
    V1, p = chain
    dh = p.get_val('delta_h')
    for v, book in FIG_5_6_DH.items():
        ratio = dh[V1 == v][0] / book
        assert 1.0 < ratio < 1.7, v


# ---------------- derivatives ----------------

@pytest.mark.parametrize('comp, inputs', [
    (DescentSpeedStencilComp(target='min_rate'), {'RD_nodes': [27.0, 26.5, 26.8]}),
    (DescentSpeedStencilComp(target='min_angle'), {'RD_nodes': [27.0, 26.5, 26.8]}),
    (SpeedNodesComp(num_nodes=3), {'V_1': [100.0, 120.0, 150.0]}),
    (ChainSplitComp(num_nodes=3), {'P_nodes': [1.0, 2.0, 3.0, 4.0]}),
])
def test_partials(comp, inputs):
    p = om.Problem()
    p.model.add_subsystem('c', comp, promotes=['*'])
    p.setup(force_alloc_complex=True)
    _set(p, inputs)
    p.run_model()
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-9, rtol=1e-9)


def test_best_speed_total_derivative_wrt_weight():
    """dV_min_rate/dGW through the nested Newton, against finite differences."""
    p = om.Problem()
    p.model.add_subsystem('g', BestAutorotationSpeedGroup(target='min_rate'), promotes=['*'])
    p.setup()
    _set(p, REF)
    p.run_model()
    J = p.compute_totals(of=['V'], wrt=['GW'])[('V', 'GW')][0, 0]
    V0 = p.get_val('V')[0]
    p.set_val('GW', 20200.0)
    p.run_model()
    fd = (p.get_val('V')[0] - V0) / 200.0
    assert_near_equal(J, fd, 0.02)
