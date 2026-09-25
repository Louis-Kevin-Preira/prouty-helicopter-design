"""C5-6 -- chart-calibrated stall torque increment (Chapter 3 charts pp. 258-266)."""
import importlib.util
import pathlib

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.performance import ForwardFlightPowerGroup
from prouty.performance.stall_torque_increment_comp import (StallTorqueIncrementComp,
                                                           build_table, CT_GRID)


def _comp(**vals):
    n = len(vals['mu'])
    p = om.Problem(reports=False)
    p.model.add_subsystem('s', StallTorqueIncrementComp(num_nodes=n), promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in vals.items():
        p.set_val(k, v)
    p.run_model()
    return p


def test_table_is_flat_below_stall():
    """Below C_T/sigma = 0.065 the increment stays under 0.0007 on every plate (the
    negative residuals of the high-collective curves, down to -0.0019, are cut by the
    floor at 0); at 0.10 it exceeds 0.0015 everywhere from mu = 0.30 up."""
    t = build_table()
    assert np.max(t[:, CT_GRID <= 0.065, :]) < 0.0007
    assert np.all(t[2:, CT_GRID == 0.1, :] > 0.0015)


def test_chart_values_at_mu_030():
    """Chart rotor (theta_1 = -5 deg): 14 deg curve at C_T/sigma = 0.09 and 0.10 gives
    0.0013 and 0.0034 (data/stall_increment.json); the component returns them."""
    import json
    d = json.load(open(pathlib.Path(__file__).parents[2] / 'src/prouty/special_performance/data/stall_increment.json'))['0.30']['14']
    c, x, v = map(np.array, (d['CT_sigma'], d['X'], d['dCQ_stall']))
    cts = np.array([0.09, 0.10])
    X = np.interp(cts, c, x)
    lam = -X * 0.3 ** 3 / (2.0 * cts)
    p = _comp(mu=[0.3, 0.3], CT_sigma=cts, lambda_p=lam, theta_1=np.deg2rad(-5.0))
    assert_near_equal(p.get_val('dCQ_sigma_stall'), np.interp(cts, c, v), 0.12)


def test_twist_and_stall_angle_shift():
    p = _comp(mu=[0.3], CT_sigma=[0.1], lambda_p=[-0.03], theta_1=np.deg2rad(-10.0),
              d_alpha_stall=np.deg2rad(1.0))
    assert_near_equal(p.get_val('CT_sigma_eff'), 0.1 - 0.015 - 6.0 / 6.0 * np.deg2rad(1.0), 1e-12)


def test_partials_against_fd():
    p = _comp(mu=[0.25, 0.32, 0.38], CT_sigma=[0.09, 0.1, 0.105], lambda_p=[-0.05, -0.03, -0.02],
              d_alpha_stall=np.deg2rad(0.5))
    assert_check_partials(p.check_partials(method='fd', form='central', step=1e-7,
                                           compact_print=True, out_stream=None),
                          atol=2e-4, rtol=5e-3)


@pytest.fixture(scope='module')
def ch4():
    path = pathlib.Path(__file__).parent / 'test_chapter4.py'
    spec = importlib.util.spec_from_file_location('ch4_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _power(ch4, W, stall):
    p = om.Problem(reports=False)
    p.model.add_subsystem('ff', ForwardFlightPowerGroup(stall=stall), promotes=['*'])
    p.setup()
    for k, v in {**ch4.REF_ROTOR, **ch4.EXAMPLE_DESIGN,
                 **dict(load_elec=2200.0, flow_hyd=1.3, p_hyd=3000.0)}.items():
        p.set_val(k, v)
    p.set_val('GW', W)
    p.set_val('V', 115 * 1.6878)
    p.set_val('alpha_F', np.deg2rad(-5.0))
    p.set_val('CT_sigma', 0.085 * W / 20000)
    p.run_model()
    return p


def test_c5_6_power_at_115_kt(ch4):
    """115 kt: +34 hp at 20,000 lb (C_T/sigma_eff = 0.071), +316 hp at 24,000 lb
    (0.088): the engine power goes from 1,301 to 1,620 hp."""
    for W, lo, hi in ((20000.0, 0.0, 60.0), (24000.0, 250.0, 380.0)):
        on, off = _power(ch4, W, True), _power(ch4, W, False)
        dP = on.get_val('P_req')[0] - off.get_val('P_req')[0]
        assert lo <= dP <= hi
        assert_near_equal(on.get_val('hp_M'), off.get_val('hp_M'), 1e-9)
