"""Chapter 5, G1 -- Turns and pullups, pp. 340-346."""
import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.special_performance import (LoadFactorComp, TurnKinematicsComp,
                                        TurnCyclicReliefComp, TurnEnergyPowerComp,
                                        ThrustCapabilityComp, TurnsPullupsGroup)

KT = 1.6878  # ft/s per knot


def _run(comp, **inputs):
    p = om.Problem()
    p.model.add_subsystem('c', comp, promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in inputs.items():
        p.set_val(k, v)
    p.run_model()
    return p


# ---------- internal consistency ----------

def test_load_factor_modes_agree_in_a_steady_turn():
    V, n = 115 * KT, 1.2
    kin = _run(TurnKinematicsComp(), V=V, n=n)
    phi, w, q = (kin.get_val(k)[0] for k in ('phi', 'omega', 'theta_dot'))
    assert_near_equal(_run(LoadFactorComp(mode='bank'), phi=phi).get_val('n'), n, 1e-12)
    assert_near_equal(_run(LoadFactorComp(mode='turn_rate'), V=V, omega=w).get_val('n'), n, 1e-12)
    assert_near_equal(_run(LoadFactorComp(mode='pitch_rate'), V=V, theta_dot=q).get_val('n'), n, 1e-12)
    # theta_dot = omega sin(phi), R*omega = V   (pp. 340-342)
    assert_near_equal(q, w * np.sin(phi), 1e-12)
    assert_near_equal(kin.get_val('R_turn') * w, V, 1e-12)


def test_pullup_and_pushover():
    V = 100 * KT
    p = _run(LoadFactorComp(mode='pullup'), V=V, theta_dot=0.1)
    assert_near_equal(p.get_val('n'), 1 + V * 0.1 / 32.2, 1e-12)
    R = V ** 2 / 32.2 / 0.5   # radius giving n = 0.5
    assert_near_equal(_run(LoadFactorComp(mode='pushover'), V=V, R_pushover=R).get_val('n'), 0.5, 1e-12)


def test_energy_power_book_option_is_twice_coherent():
    args = dict(GW=20000., n=1.5, V_avg=107.5 * KT, delta_V=15 * KT, delta_h=50.)
    c = _run(TurnEnergyPowerComp(turn_time='coherent'), **args)
    b = _run(TurnEnergyPowerComp(turn_time='book'), **args)
    for k in ('dhp_dV', 'dhp_dh'):
        assert_near_equal(b.get_val(k), 2 * c.get_val(k), 1e-12)
    assert_near_equal(c.get_val('t_180'), 2 * b.get_val('t_180'), 1e-12)


# ---------- book anchors ----------

def test_anchor_p343_energy_power_book():
    """1.5 g, 180 deg turn 115 -> 100 kt, 50 ft lost: 656 hp and 230 hp (p. 343)."""
    p = _run(TurnEnergyPowerComp(turn_time='book'),
             GW=20000., n=1.5, V_avg=107.5 * KT, delta_V=15 * KT, delta_h=50.)
    assert_near_equal(p.get_val('dhp_dV', units='hp'), 656., 0.005)
    assert_near_equal(p.get_val('dhp_dh', units='hp'), 230., 0.005)


def test_anchor_p342_pitch_rate_and_relief():
    """115 kt, n = 1.2: printed 0.08 rad/s and 0.2 deg (p. 342) -> C5-4, C5-5.

    The relations give 0.061 rad/s and 0.32 deg (gamma = 8.1, Omega = 21.67, App. A).
    """
    q = _run(TurnKinematicsComp(), V=115 * KT, n=1.2).get_val('theta_dot')[0]
    assert_near_equal(q, 0.0608, 0.005)
    dB1 = _run(TurnCyclicReliefComp(), theta_dot=q, lock_number=8.1, Omega=21.67).get_val('delta_B1', units='deg')
    assert_near_equal(dB1, -0.317, 0.01)


def test_anchor_fig52_transient_near_017():
    """Transient boundary 'in the neighborhood of CT/sigma = 0.17' (p. 344)."""
    p = _run(ThrustCapabilityComp(num_nodes=3, boundary='transient'),
             mu=[0.1, 0.2, 0.3], band_fraction=[0., 0., 0.])
    assert np.all(np.abs(p.get_val('CT_sigma_limit') - 0.17) < 0.006)


def test_fig52_level_band_edges_at_mu_050():
    """Right-hand labels: Low Drag ~0.077, High Drag ~0.038 at mu = 0.5 (p. 345)."""
    p = _run(ThrustCapabilityComp(num_nodes=2, boundary='level'),
             mu=[0.5, 0.5], band_fraction=[0., 1.])
    assert_near_equal(p.get_val('CT_sigma_limit'), [0.077, 0.038], 1e-6)


def test_fig52_band_ordering():
    mu = np.linspace(0.05, 0.5, 10)
    vals = {}
    for b in ('transient', 'steady_turn', 'level'):
        p = _run(ThrustCapabilityComp(num_nodes=10, boundary=b), mu=mu)
        vals[b] = p.get_val('CT_sigma_limit')
    assert np.all(vals['transient'] >= vals['steady_turn'])
    assert np.all(vals['steady_turn'] >= vals['level'])


# ---------- group ----------

@pytest.mark.parametrize('mode', ['bank', 'turn_rate', 'pitch_rate', 'pullup', 'pushover'])
def test_group_runs_all_modes(mode):
    p = om.Problem()
    p.model.add_subsystem('g', TurnsPullupsGroup(num_nodes=2, load_factor_mode=mode), promotes=['*'])
    p.setup(force_alloc_complex=True)
    defaults = {'phi': [0.5, 0.6], 'V': [194., 170.], 'omega': [0.15, 0.2],
                'theta_dot': [0.06, 0.08], 'R_pushover': [3000., 4000.],
                'V_avg': [180., 160.], 'delta_V': [25., 20.], 'delta_h': [50., 40.],
                'mu': [0.3, 0.25], 'CW_sigma': [0.08, 0.08]}
    for k, v in defaults.items():
        try:
            p.set_val(k, v)
        except KeyError:
            pass
    p.run_model()
    assert np.all(np.isfinite(p.get_val('n_margin')))
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-8, rtol=1e-8)


# ---------- derivatives ----------

@pytest.mark.parametrize('comp, inputs', [
    (LoadFactorComp(num_nodes=2, mode='bank'), {'phi': [0.3, 0.7]}),
    (LoadFactorComp(num_nodes=2, mode='turn_rate'), {'V': [190., 150.], 'omega': [0.1, 0.2]}),
    (LoadFactorComp(num_nodes=2, mode='pitch_rate'), {'V': [190., 150.], 'theta_dot': [0.06, 0.1]}),
    (LoadFactorComp(num_nodes=2, mode='pullup'), {'V': [190., 150.], 'theta_dot': [0.06, 0.1]}),
    (LoadFactorComp(num_nodes=2, mode='pushover'), {'V': [190., 150.], 'R_pushover': [3000., 5000.]}),
    (TurnKinematicsComp(num_nodes=2), {'V': [190., 150.], 'n': [1.2, 1.8]}),
    (TurnCyclicReliefComp(num_nodes=2), {'theta_dot': [0.06, 0.1]}),
    (TurnEnergyPowerComp(num_nodes=2), {'n': [1.3, 1.6], 'V_avg': [180., 150.],
                                        'delta_V': [25., 10.], 'delta_h': [50., 20.]}),
    (ThrustCapabilityComp(num_nodes=3), {'mu': [0.03, 0.22, 0.47], 'band_fraction': [0.2, 0.5, 0.9],
                                         'n': [1.2, 1.5, 1.1]}),
])
def test_partials(comp, inputs):
    p = _run(comp, **inputs)
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ---------- steady turn power through Chapter 4 (p. 343, C5-6) ----------

@pytest.fixture(scope='module')
def ch4():
    """Chapter 4 test module: example helicopter inputs of G7 (REF_ROTOR, EXAMPLE_DESIGN)."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).parents[1] / 'performance' / 'test_chapter4.py'
    spec = importlib.util.spec_from_file_location('ch4_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _turn_power(ch4, n, V_kt=115.0, stall=True):
    from prouty.special_performance import SteadyTurnPowerGroup
    p = om.Problem()
    p.model.add_subsystem('turn', SteadyTurnPowerGroup(stall=stall), promotes=['*'])
    p.setup()
    for name, val in {**ch4.REF_ROTOR, **ch4.EXAMPLE_DESIGN,
                      **dict(load_elec=2200.0, flow_hyd=1.3, p_hyd=3000.0)}.items():
        p.set_val(name, val)
    p.set_val('n', n)
    p.set_val('V', V_kt * KT)
    p.set_val('alpha_F', np.deg2rad(-6.0))
    p.set_val('CT_sigma', 0.085 * n)
    p.run_model()
    return p


def test_turn_power_is_level_power_at_effective_weight(ch4):
    """p. 343: power in the turn = level power at n*GW (Chapter 4 run at 24,000 lb)."""
    turn = _turn_power(ch4, 1.2, stall=False).get_val('P_req', units='hp')[0]
    level = ch4._ff_power(115.0, GW=24000.0, CT_sigma=0.102).get_val('P_req', units='hp')[0]
    assert_near_equal(turn, level, 1e-4)   # trim solver tolerance


def test_turn_power_below_print_c5_6(ch4):
    """C5-6: 1.2 g at 115 kt, printed 3,170 hp against 1,470 hp level (Fig. 4.38, p. 343).

    Without the stall increment the chain rises by 8 % (1,206 -> 1,301 hp); with it
    (default) by 34 % (1,206 -> 1,620 hp: +316 hp at the rotor at C_T/sigma_eff = 0.088).
    The printed 116 % follows the 24,000 lb curve of Fig. 4.38 on the upper stall
    limit; the p. 230 twist shift keeps the example (-10 deg) 0.015 further from stall.
    """
    p1 = _turn_power(ch4, 1.0).get_val('P_req', units='hp')[0]
    p12 = _turn_power(ch4, 1.2).get_val('P_req', units='hp')[0]
    p12_nostall = _turn_power(ch4, 1.2, stall=False).get_val('P_req', units='hp')[0]
    assert 1.0 < p12_nostall / p1 < 1.12
    assert 1.28 < p12 / p1 < 1.40
    assert 250.0 < p12 - p12_nostall < 380.0
    assert p12 < 3170.0
