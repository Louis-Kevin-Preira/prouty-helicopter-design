"""Chapter 5 links to Chapter 4: hover energy times (G2e/G2f), multiengine
sink speed (G2d), takeoff capability (G5)."""
import importlib.util
import pathlib

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.special_performance import (HoverEnergyGroup, MultiEngineSinkGroup,
                                        LinearAccelerationComp, TakeoffCapabilityGroup)
from prouty.special_performance.book_figures import EXAMPLE_POWER, EXAMPLE_ROTOR


@pytest.fixture(scope='module')
def ch4():
    path = pathlib.Path(__file__).parents[1] / 'performance' / 'test_chapter4.py'
    spec = importlib.util.spec_from_file_location('ch4_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_hover(p, prefix, ch4):
    for k, v in ch4.EXAMPLE_HOVER.items():
        if k not in ('altitude', 'GW'):
            p.set_val(prefix + k, v)
    p.set_val(prefix + 'altitude', 0.0)
    p.set_val(prefix + 'theta_0', 17.0)
    p.set_val(prefix + 'rotor_ref.theta_0', 13.0)
    p.set_val(prefix + 'tr_theta_0', 15.0)


def test_hover_energy_times_c5_9(ch4):
    """C5-9: with the Chapter 4 engine hover power (2,307 hp) and the main rotor
    maximum 0.167, t_equiv = 0.82 s (printed 0.8, p. 363) and the flare time
    1.09 s (printed 1.25, p. 362). Both printed times cannot come from one power."""
    p = om.Problem(reports=False)
    p.model.add_subsystem('g', HoverEnergyGroup(), promotes=['*'])
    p.setup()
    _set_hover(p, 'hover.', ch4)
    p.set_val('J', 11735.0)
    p.run_model()
    assert_near_equal(p.get_val('hover.P_req', units='hp'), 2307.4, 1e-3)
    assert_near_equal(p.get_val('t_equiv'), 0.8, 0.04)
    assert 0.85 < p.get_val('dt')[0] / 1.25 < 0.9


def _sink(td, P_avail):
    p = om.Problem(reports=False)
    p.model.add_subsystem('g', MultiEngineSinkGroup(time_delay=td), promotes=['*'])
    p.setup()
    for k, v in {**{k: v for k, v in EXAMPLE_ROTOR.items() if k != 'GW'}, **EXAMPLE_POWER}.items():
        p.set_val('power.' + k, v)
    p.set_val('power.CT_sigma', 0.085)
    p.set_val('power.alpha_F', np.deg2rad(-15.0) * np.ones(3))
    for k, v in dict(P_hover=2307.0, P_avail=P_avail, V_LG=6.0, h_lo=30.0).items():
        p.set_val(k, v)
    p.run_model()
    return p


@pytest.mark.parametrize('td, k', [('faa', 0.5), ('military', 1.0)])
def test_multi_engine_sink_speed(td, k):
    """Example twin, one engine left (about 2,000 hp), sea level: the remaining power
    holds 6 ft/s of sink at a few knots, so V_CR is small -- Figure 5.10 shows no
    single-engine envelope at sea level."""
    p = _sink(td, 2000.0)
    assert_near_equal(p.get_val('RD'), 6.0, 1e-8)
    assert_near_equal(p.get_val('V_CR'), k * p.get_val('V_sink'), 1e-12)
    assert p.get_val('V_sink')[0] < 12.0
    assert_near_equal(p.get_val('h_CR'), 50.0, 1e-6)
    assert _sink(td, 1900.0).get_val('V_sink')[0] > p.get_val('V_sink')[0]


def test_takeoff_capability(ch4):
    """IGE thrust at Z/D = 0.25 (P_req = P_rating) and the linear law of p. 367 at
    28,000 lb: T_max_IGE = 31,574 lb, x_ddot_HIGE = 16.8 ft/s^2, V_max = 214 kt."""
    p = om.Problem(reports=False)
    p.model.add_subsystem('g', TakeoffCapabilityGroup(), promotes=['*'])
    p.setup()
    _set_hover(p, 'ige.', ch4)
    p.set_val('ige.rotor_height_D', 0.25)
    p.set_val('forward.sigma', 0.0849)
    p.set_val('forward.cd_bar', 0.01)
    p.set_val('forward.R', 30.0)
    p.run_model()
    assert_near_equal(p.get_val('ige.P_req'), p.get_val('ige.P_rating'), 1e-8)
    assert_near_equal(p.get_val('T_max_IGE'), 31574.0, 1e-3)
    assert_near_equal(p.get_val('acc_0'), 16.78, 2e-3)
    assert_near_equal(p.get_val('V_max', units='kn'), 214.4, 2e-3)


def test_linear_acceleration_partials():
    p = om.Problem(reports=False)
    p.model.add_subsystem('c', LinearAccelerationComp(), promotes=['*'])
    p.setup(force_alloc_complex=True)
    p.run_model()
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-9, rtol=1e-9)
