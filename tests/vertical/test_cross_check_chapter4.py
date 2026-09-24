"""Chapter 2 against Chapter 4: the vertical climb of Figure 2.4 against Figure 4.36 (C2-6).

The same equation (Ch. 2 p. 98 = Ch. 4 p. 314), the same helicopter, the same
author. Chapter 4 evaluates it at 20,000 lb with its own vertical drag and tail
rotor (C4-28), and reproduces Figure 4.36; Figure 2.4 is the one that departs.
The Chapter 4 chain is run exactly as validated in tests/performance.
"""

import importlib.util
import pathlib
import warnings

import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_near_equal

from prouty.vertical import ClimbPowerGroup

FIG_4_36_SL = 3270.0     # ft/min, 20,000 lb, takeoff power, sea level, standard day (p. 317)
FIG_2_4 = np.array([[251, 100], [502, 200], [934, 400], [1498, 700], [2000, 1000],
                    [2606, 1400], [3136, 1800], [3505, 2100], [3854, 2400]], dtype=float)


@pytest.fixture(scope='module')
def ch4():
    """VerticalClimbGroup of Chapter 4 at sea level, standard day (C4-28)."""
    path = pathlib.Path(__file__).parents[1] / 'performance' / 'test_chapter4.py'
    spec = importlib.util.spec_from_file_location('ch4_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        p = module._climb_problem()
    return {name: p.get_val(name)[0] for name in
            ('V_c', 'P_excess', 'T_target', 'Dv_GW', 'v_hov_T', 'A', 'l_T', 'GW')}


def _dP(state, V_c, GW=None, T=None, Dv_GW=None):
    """Power increment of G2 with the Chapter 4 state, optionally changed."""
    p = om.Problem()
    p.model.add_subsystem('g2', ClimbPowerGroup(), promotes=['*'])
    p.setup()
    p.set_val('GW', state['GW'] if GW is None else GW, units='lbf')
    p.set_val('T', state['T_target'] if T is None else T, units='lbf')
    p.set_val('Dv_GW', state['Dv_GW'] if Dv_GW is None else Dv_GW)
    p.set_val('V_c', V_c, units='ft/s')
    p.set_val('A', state['A'], units='ft**2')
    p.set_val('R', 30.0, units='ft')
    p.set_val('l_T', state['l_T'], units='ft')
    p.set_val('v_hov_T', state['v_hov_T'], units='ft/s')
    p.run_model()
    return p.get_val('dP', units='hp')[0]


def _rate_of_climb(state, **change):
    """ft/min at which G2 uses the Chapter 4 excess power."""
    lo, hi = 0.0, 120.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _dP(state, mid, **change) < state['P_excess'] else (lo, mid)
    return 30.0 * (lo + hi)


def test_g2_is_the_chapter4_climb_equation(ch4):
    """At the Chapter 4 solution, G2 returns the Chapter 4 excess power."""
    assert_near_equal(_dP(ch4, ch4['V_c']), ch4['P_excess'], 1e-6)


def test_chapter4_reproduces_figure_4_36(ch4):
    assert_near_equal(ch4['V_c'] * 60.0, FIG_4_36_SL, 0.01)


def test_figure_2_4_contradicts_figure_4_36(ch4):
    """Read backwards at the same excess power, Figure 2.4 gives about 3,100 ft/min, 5 % under 4.36."""
    rc_fig_2_4 = np.interp(ch4['P_excess'], FIG_2_4[:, 1], FIG_2_4[:, 0])
    assert rc_fig_2_4 < 0.96 * FIG_4_36_SL
    assert_near_equal(rc_fig_2_4, 3095.0, 0.01)


@pytest.mark.parametrize('candidate', ['Dv_GW = 0.06', 'GW = 21,900 lb'])
def test_c2_6_candidates_contradict_figure_4_36(ch4, candidate):
    """Either explanation of Figure 2.4 would put Figure 4.36 6-7 % low."""
    if candidate == 'Dv_GW = 0.06':
        change = {'Dv_GW': 0.06}
    else:   # same download ratio as Chapter 4
        change = {'GW': 21900.0, 'T': 21900.0 * ch4['T_target'] / ch4['GW']}
    rc = _rate_of_climb(ch4, **change)
    assert rc < 0.95 * FIG_4_36_SL
