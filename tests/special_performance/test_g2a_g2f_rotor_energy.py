"""Chapter 5, G2a -- rotor speed decay (pp. 348-350); G2f -- autorotative indices (pp. 363-364)."""
import numpy as np
import openmdao.api as om
import pytest
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from prouty.special_performance import (DriveInertiaComp, KineticEnergyTimeComp,
                                        RotorSpeedDecayComp, RotorSpeedDecayGroup,
                                        EquivalentHoverTimeComp, AutorotativeIndexComp,
                                        AutorotativeIndicesGroup)

J_BOOK = 11735.0
DL_EXAMPLE = 20000.0 / (np.pi * 30.0 ** 2)


def _run(system, **inputs):
    p = om.Problem()
    p.model.add_subsystem('s', system, promotes=['*'])
    p.setup(force_alloc_complex=True)
    for k, v in inputs.items():
        p.set_val(k, v)
    p.run_model()
    return p


# ---------- book anchors ----------

def test_anchor_p348_inertia_book_and_coherent():
    """C5-1: printed 11,735 slug ft^2; squared speed ratio gives 12,152."""
    assert_near_equal(_run(DriveInertiaComp(inertia_ratio='book')).get_val('J'), J_BOOK, 1e-4)
    assert_near_equal(_run(DriveInertiaComp()).get_val('J'),
                      11600.0 + (100.0 / 21.67) ** 2 * 25.0 + 20.0, 1e-12)


def test_anchor_p349_kinetic_energy_time():
    """t_KE = 1.2 s at full power, two 2,000 hp engines (p. 349, Appendix A)."""
    t = _run(KineticEnergyTimeComp(), J=J_BOOK, P_0=4000.0).get_val('t_KE')
    assert_near_equal(t, 1.2, 0.05)     # 1.25 s, printed rounded


@pytest.mark.parametrize('f, drop', [(1.0, 0.30), (0.5, 0.17)])
def test_anchor_p350_decay_first_second(f, drop):
    """30 % in the first second, both engines; 17 % with one engine out (p. 350)."""
    r = _run(RotorSpeedDecayComp(), t=1.0, t_KE=1.2, power_loss_fraction=f).get_val('Omega_ratio')
    assert_near_equal(1.0 - r, drop, 0.03)


def test_anchor_p363_autorotative_index():
    """Example helicopter AI = 39 ft^3/lb (p. 363)."""
    ai = _run(AutorotativeIndexComp(), J=J_BOOK, DL=DL_EXAMPLE).get_val('AI')
    assert_near_equal(ai, 39.0, 0.01)


# ---------- consistency ----------

def test_decay_closed_form_integrates_the_ode():
    """Omega_dot = -f Q_0 (Omega/Omega_0)^2 / J integrated numerically (p. 348)."""
    tk, f = 1.3, 0.8
    t = np.linspace(0.0, 3.0, 7)
    r = _run(RotorSpeedDecayComp(num_nodes=7), t=t, t_KE=tk, power_loss_fraction=f).get_val('Omega_ratio')
    # nondimensional: d(ratio)/dt = -f ratio^2 / (2 t_KE)
    ts = np.linspace(0.0, 3.0, 30001)
    y = np.empty_like(ts); y[0] = 1.0
    h = ts[1] - ts[0]
    for i in range(len(ts) - 1):
        k1 = -f * y[i] ** 2 / (2 * tk)
        k2 = -f * (y[i] + h * k1) ** 2 / (2 * tk)
        y[i + 1] = y[i] + 0.5 * h * (k1 + k2)
    assert_near_equal(r, np.interp(t, ts, y), 1e-7)


def test_equivalent_hover_time_matches_kinetic_energy_time():
    """t_equiv = t_KE (1 - C_W/sigma / (0.8 C_T/sigma max)) at P_0 = P_OGE."""
    te = _run(EquivalentHoverTimeComp(), J=J_BOOK, CW_sigma=0.083, CT_sigma_max=0.155,
              P_OGE=2000.0).get_val('t_equiv')
    tk = _run(KineticEnergyTimeComp(), J=J_BOOK, P_0=2000.0).get_val('t_KE')
    assert_near_equal(te, tk * (1 - 0.083 / (0.8 * 0.155)), 1e-12)


def test_groups_connect_inertia():
    p = om.Problem()
    p.model.add_subsystem('decay', RotorSpeedDecayGroup(num_nodes=3, inertia_ratio='book'))
    p.model.add_subsystem('indices', AutorotativeIndicesGroup())
    p.model.connect('decay.J', 'indices.J')
    p.setup(force_alloc_complex=True)
    p.set_val('decay.t', [0.5, 1.0, 2.0])
    p.set_val('decay.P_0', 4000.0)
    p.set_val('indices.DL', DL_EXAMPLE)
    p.run_model()
    assert_near_equal(p.get_val('indices.AI'), 39.0, 0.01)
    assert np.all(np.diff(p.get_val('decay.Omega_ratio')) < 0)
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-9, rtol=1e-9)


# ---------- derivatives ----------

@pytest.mark.parametrize('comp, inputs', [
    (DriveInertiaComp(), {}),
    (DriveInertiaComp(inertia_ratio='book'), {}),
    (KineticEnergyTimeComp(num_nodes=2), {'P_0': [4000.0, 2500.0]}),
    (RotorSpeedDecayComp(num_nodes=3), {'t': [0.3, 1.0, 2.5], 'power_loss_fraction': 0.6}),
    (EquivalentHoverTimeComp(), {}),
    (AutorotativeIndexComp(), {'rho_ratio': 0.86}),
])
def test_partials(comp, inputs):
    p = _run(comp, **inputs)
    assert_check_partials(p.check_partials(method='cs', compact_print=True, out_stream=None),
                          atol=1e-9, rtol=1e-9)
